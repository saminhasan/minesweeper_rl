from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TypeAlias, cast

import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from game_engine import Minesweeper, game_mode

DIFFICULTY = "xtreme"
NUM_SAMPLES = 2**24
SAFE_REVEAL = False
SEED = 123
N_JOBS = 12
OUT = f"data/{DIFFICULTY}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_minesweeper_{DIFFICULTY}_{NUM_SAMPLES}_{SEED}.npz"
SAVE_DATASET = False

MAX_SAMPLES_PER_GAME = 4
MIN_REVEALED_FRAC = 0.05
MAX_REVEALED_FRAC = 0.95

Array3D: TypeAlias = np.ndarray
ChunkResult: TypeAlias = tuple[Array3D, Array3D, Array3D]


@contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Iterator[tqdm]:
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: object, **kwargs: object) -> object:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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
    return num_samples * rows * cols * 3 * 4


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
    print("arrays=x,y,mask float32")
    print(f"estimated uncompressed memory={format_bytes(estimated_bytes)} ({estimated_bytes} bytes)")
    print(f"joblib workers={N_JOBS}")
    print()

    response = input("Proceed? [Y/n]: ").strip().lower()
    return response in ("", "y", "yes")


def split_samples(total: int, n_jobs: int) -> list[int]:
    base = total // n_jobs
    rem = total % n_jobs
    return [base + (1 if i < rem else 0) for i in range(n_jobs)]


def generate_dataset_chunk(
    difficulty: str,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
) -> ChunkResult:
    rng = np.random.default_rng(seed)

    rows = game_mode[difficulty]["rows"]
    cols = game_mode[difficulty]["columns"]
    covered = float(Minesweeper(difficulty=difficulty, seed=0).states.COVERED)

    x_arr = np.empty((num_samples, rows, cols), dtype=np.float32)
    y_arr = np.empty((num_samples, rows, cols), dtype=np.float32)
    mask_arr = np.empty((num_samples, rows, cols), dtype=np.float32)

    sample_idx = 0

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

            if should_keep_sample(board, revealed, MAX_SAMPLES_PER_GAME, kept_from_game):
                x_arr[sample_idx] = board.get_input()
                y_arr[sample_idx] = board.get_output()
                mask_arr[sample_idx] = x_arr[sample_idx] == covered
                sample_idx += 1
                kept_from_game += 1

            reveal()

    return x_arr, y_arr, mask_arr


def generate_dataset_parallel(
    difficulty: str,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
    n_jobs: int,
) -> ChunkResult:
    chunk_sizes = split_samples(num_samples, n_jobs)
    active_chunk_count = sum(1 for size in chunk_sizes if size > 0)

    with tqdm_joblib(
        tqdm(total=active_chunk_count, desc="Generating chunks", unit="chunk")
    ):
        results_raw = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(generate_dataset_chunk)(
                difficulty=difficulty,
                num_samples=chunk_sizes[i],
                safe_reveal=safe_reveal,
                seed=seed + i,
            )
            for i in range(n_jobs)
            if chunk_sizes[i] > 0
        )
    results = cast(list[ChunkResult], [r for r in results_raw if r is not None])

    if not results:
        rows = game_mode[difficulty]["rows"]
        cols = game_mode[difficulty]["columns"]
        empty = np.empty((0, rows, cols), dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    x_arr = np.concatenate([r[0] for r in results], axis=0)
    y_arr = np.concatenate([r[1] for r in results], axis=0)
    mask_arr = np.concatenate([r[2] for r in results], axis=0)
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


if __name__ == "__main__":
    main()