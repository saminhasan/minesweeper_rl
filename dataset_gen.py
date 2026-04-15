from datetime import datetime
import gc
from multiprocessing import Manager
from pathlib import Path
from threading import Thread
from time import sleep
from typing import TypeAlias

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from game_engine import Minesweeper, game_mode

DIFFICULTY = "hard"
NUM_SAMPLES = 2**24
SAFE_REVEAL = False
SEED = 123
N_JOBS = 8
OUT = f"data/{DIFFICULTY}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_minesweeper_{DIFFICULTY}_{NUM_SAMPLES}_{SEED}.npz"
SAVE_DATASET = True

MAX_SAMPLES_PER_GAME = 4
MIN_REVEALED_FRAC = 0.05
MAX_REVEALED_FRAC = 0.95

Array3D: TypeAlias = np.ndarray
ChunkResult: TypeAlias = tuple[Array3D, Array3D, Array3D]


def count_revealed(board: Minesweeper) -> int:
    return int(np.count_nonzero(board.state == board.states.UNCOVERED))


def should_keep_sample(
    board: Minesweeper,
    revealed: int,
    max_samples_from_game: int,
    kept_from_game: int,
) -> bool:
    if kept_from_game >= max_samples_from_game:
        return False

    total_cells = board.n_rows * board.n_cols
    revealed_frac = revealed / total_cells

    if revealed_frac < MIN_REVEALED_FRAC or revealed_frac > MAX_REVEALED_FRAC:
        return False

    if int(np.count_nonzero(board.get_frontier_cells())) == 0:
        return False

    return True


def estimate_memory_bytes(num_samples: int, rows: int, cols: int) -> int:
    bytes_per_cell = np.dtype(np.float32).itemsize * 2 + np.dtype(np.bool_).itemsize
    return num_samples * rows * cols * bytes_per_cell


def format_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(n)
    unit = units[0]
    for next_unit in units[1:]:
        if size < 1024.0:
            break
        size /= 1024.0
        unit = next_unit
    return f"{size:.2f} {unit}"


def confirm_proceed(num_samples: int, rows: int, cols: int) -> bool:
    estimated_bytes = estimate_memory_bytes(num_samples, rows, cols)
    print("Dataset allocation estimate")
    print(f"difficulty={DIFFICULTY}")
    print(f"shape=({num_samples}, {rows}, {cols}) per array")
    print("arrays=x,y float32; mask bool")
    print(f"estimated uncompressed dataset size={format_bytes(estimated_bytes)} ({estimated_bytes} bytes)")
    print("generation mode=disk-backed memmap (low RAM)")
    print(f"joblib workers={N_JOBS}")
    print()

    response = input("Proceed? [Y/n]: ").strip().lower()
    return response in ("", "y", "yes")


def split_samples(total: int, n_jobs: int) -> list[int]:
    base = total // n_jobs
    rem = total % n_jobs
    return [base + (1 if i < rem else 0) for i in range(n_jobs)]


def chunk_starts(chunk_sizes: list[int]) -> list[int]:
    starts: list[int] = []
    cur = 0
    for size in chunk_sizes:
        starts.append(cur)
        cur += size
    return starts


def memmap_paths(out_path: Path) -> tuple[Path, Path, Path]:
    tmp_base = out_path.with_suffix("")
    return (
        tmp_base.with_name(tmp_base.name + "_x.tmp.npy"),
        tmp_base.with_name(tmp_base.name + "_y.tmp.npy"),
        tmp_base.with_name(tmp_base.name + "_mask.tmp.npy"),
    )


def progress_monitor(counter, total: int) -> None:
    last = 0
    with tqdm(total=total, desc="Generating samples", unit="sample", mininterval=0.2) as pbar:
        while last < total:
            cur = counter.value
            if cur > last:
                pbar.update(cur - last)
                last = cur
            else:
                sleep(0.1)


def generate_dataset_chunk_to_memmap(
    difficulty: str,
    start_idx: int,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
    x_path: str,
    y_path: str,
    mask_path: str,
    total_samples: int,
    counter,
    lock,
) -> None:
    rng = np.random.default_rng(seed)

    rows = game_mode[difficulty]["rows"]
    cols = game_mode[difficulty]["columns"]
    covered = float(Minesweeper(difficulty=difficulty, seed=0).states.COVERED)

    x_arr = np.lib.format.open_memmap(
        x_path,
        mode="r+",
        dtype=np.float32,
        shape=(total_samples, rows, cols),
    )
    y_arr = np.lib.format.open_memmap(
        y_path,
        mode="r+",
        dtype=np.float32,
        shape=(total_samples, rows, cols),
    )
    mask_arr = np.lib.format.open_memmap(
        mask_path,
        mode="r+",
        dtype=np.bool_,
        shape=(total_samples, rows, cols),
    )

    sample_idx = 0
    progress_local = 0

    while sample_idx < num_samples:
        board = Minesweeper(
            difficulty=difficulty,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

        reveal = board.random_safe_reveal if safe_reveal else board.random_reveal
        board.random_safe_reveal()

        kept_from_game = 0

        while sample_idx < num_samples and not (board.game_over or board.game_won):
            revealed = count_revealed(board)
            write_idx = start_idx + sample_idx

            if should_keep_sample(board, revealed, MAX_SAMPLES_PER_GAME, kept_from_game):
                x_arr[write_idx] = board.get_input()
                y_arr[write_idx] = board.get_output()
                mask_arr[write_idx] = x_arr[write_idx] == covered
                sample_idx += 1
                kept_from_game += 1
                progress_local += 1

                if progress_local >= 64:
                    with lock:
                        counter.value += progress_local
                    progress_local = 0

            reveal()

    if progress_local:
        with lock:
            counter.value += progress_local
    x_arr.flush()
    y_arr.flush()
    mask_arr.flush()


def generate_dataset_parallel(
    difficulty: str,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
    n_jobs: int,
    out_path: Path,
) -> ChunkResult:
    rows = game_mode[difficulty]["rows"]
    cols = game_mode[difficulty]["columns"]

    x_tmp_path, y_tmp_path, mask_tmp_path = memmap_paths(out_path)
    x_path = str(x_tmp_path)
    y_path = str(y_tmp_path)
    mask_path = str(mask_tmp_path)

    x_arr = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, rows, cols),
    )
    y_arr = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, rows, cols),
    )
    mask_arr = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(num_samples, rows, cols),
    )
    x_arr.flush()
    y_arr.flush()
    mask_arr.flush()

    del x_arr
    del y_arr
    del mask_arr

    if num_samples == 0:
        x_empty = np.lib.format.open_memmap(x_path, mode="r+", dtype=np.float32, shape=(0, rows, cols))
        y_empty = np.lib.format.open_memmap(y_path, mode="r+", dtype=np.float32, shape=(0, rows, cols))
        mask_empty = np.lib.format.open_memmap(mask_path, mode="r+", dtype=np.bool_, shape=(0, rows, cols))
        return x_empty, y_empty, mask_empty

    chunk_sizes = split_samples(num_samples, n_jobs)
    starts = chunk_starts(chunk_sizes)

    with Manager() as mgr:
        counter = mgr.Value("q", 0)
        lock = mgr.Lock()

        progress_thread = Thread(
            target=progress_monitor,
            args=(counter, num_samples),
            daemon=True,
        )
        progress_thread.start()

        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(generate_dataset_chunk_to_memmap)(
                difficulty=difficulty,
                start_idx=starts[i],
                num_samples=chunk_sizes[i],
                safe_reveal=safe_reveal,
                seed=seed + i,
                x_path=x_path,
                y_path=y_path,
                mask_path=mask_path,
                total_samples=num_samples,
                counter=counter,
                lock=lock,
            )
            for i in range(n_jobs)
            if chunk_sizes[i] > 0
        )

        progress_thread.join()
    x_arr = np.load(x_path, mmap_mode="r")
    y_arr = np.load(y_path, mmap_mode="r")
    mask_arr = np.load(mask_path, mmap_mode="r")
    return x_arr, y_arr, mask_arr


def main() -> None:
    rows = game_mode[DIFFICULTY]["rows"]
    cols = game_mode[DIFFICULTY]["columns"]

    if not confirm_proceed(NUM_SAMPLES, rows, cols):
        print("Aborted.")
        return

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_arr, y_arr, mask_arr = generate_dataset_parallel(
        difficulty=DIFFICULTY,
        num_samples=NUM_SAMPLES,
        safe_reveal=SAFE_REVEAL,
        seed=SEED,
        n_jobs=N_JOBS,
        out_path=out_path,
    )

    print("Generated dataset")
    print(f"x shape={x_arr.shape}, dtype={x_arr.dtype}")

    if SAVE_DATASET:
        np.savez_compressed(
            out_path,
            x=x_arr,
            y=y_arr,
            mask=mask_arr,
            difficulty=np.array(DIFFICULTY),
        )

        print("Saved dataset")
        print(f"path={out_path}")
        print(f"x shape={x_arr.shape}, dtype={x_arr.dtype}")
        print(f"y shape={y_arr.shape}, dtype={y_arr.dtype}")
        print(f"mask shape={mask_arr.shape}, dtype={mask_arr.dtype}")

        # On Windows, memmap-backed files can remain locked until references are released.
        del x_arr
        del y_arr
        del mask_arr
        gc.collect()

        x_tmp_path, y_tmp_path, mask_tmp_path = memmap_paths(out_path)
        locked_paths: list[Path] = []
        for tmp_path in (x_tmp_path, y_tmp_path, mask_tmp_path):
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except PermissionError:
                    locked_paths.append(tmp_path)

        if locked_paths:
            print("Temporary memmap files retained because they are still locked by the OS:")
            for p in locked_paths:
                print(f"  {p}")
        else:
            print("Removed temporary memmap files")


if __name__ == "__main__":
    main()