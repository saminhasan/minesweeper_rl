from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TypeAlias, TypedDict

import numpy as np
from tqdm import tqdm

from game_engine import Minesweeper, game_mode


DIFFICULTY = "hard" # intermediate # hard # easy
NUM_SAMPLES = 2**20
SAFE_REVEAL = True
SEED = 123

OUT_BASE_DIR = Path("data_bin")
RUN_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_minesweeper_{DIFFICULTY}_{NUM_SAMPLES}_{SEED}"
OUT_DIR = OUT_BASE_DIR / DIFFICULTY / RUN_NAME

MAX_SAMPLES_PER_GAME = 16
MIN_REVEALED_FRAC = 0.05
MAX_REVEALED_FRAC = 0.95

TARGET_DIR_SIZE_GB = 18.0
TARGET_DIR_BYTES = int(TARGET_DIR_SIZE_GB * (1024**3))
PROMPT_BEFORE_RUN = True

PAYLOAD_DTYPE = np.float64
PAYLOAD_PLANES = ("x_input", "y_prob", "covered_mask", "mine_mask")

Array2D: TypeAlias = np.ndarray


class GenerationSummary(TypedDict):
    num_samples: int
    games_simulated: int
    rows: int
    cols: int
    bytes_per_snapshot: int
    files_per_dir: int
    target_dir_bytes: int


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


def bytes_per_snapshot(rows: int, cols: int) -> int:
    bytes_per_value = np.dtype(PAYLOAD_DTYPE).itemsize
    return len(PAYLOAD_PLANES) * rows * cols * bytes_per_value


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


def files_per_dir_for_shape(rows: int, cols: int, target_dir_bytes: int = TARGET_DIR_BYTES) -> int:
    return max(1, target_dir_bytes // max(1, bytes_per_snapshot(rows, cols)))


def sample_rel_path(sample_idx: int, files_per_dir: int) -> Path:
    dir_idx = sample_idx // files_per_dir
    return Path(f"chunk_{dir_idx:06d}") / f"snap_{sample_idx:012d}.bin"


def snapshot_planes(board: Minesweeper) -> tuple[Array2D, Array2D, Array2D, Array2D]:
    x_input = board.get_input().astype(PAYLOAD_DTYPE, copy=False)
    y_prob = board.get_output().astype(PAYLOAD_DTYPE, copy=False)
    covered_mask = (board.state == board.states.COVERED).astype(PAYLOAD_DTYPE, copy=False)
    mine_mask = (board.mine_count == -1).astype(PAYLOAD_DTYPE, copy=False)
    return x_input, y_prob, covered_mask, mine_mask


def write_snapshot_payload(
    out_path: Path,
    x_input: Array2D,
    y_prob: Array2D,
    covered_mask: Array2D,
    mine_mask: Array2D,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        np.asarray(x_input, dtype=PAYLOAD_DTYPE, order="C").ravel(order="C").tofile(f)
        np.asarray(y_prob, dtype=PAYLOAD_DTYPE, order="C").ravel(order="C").tofile(f)
        np.asarray(covered_mask, dtype=PAYLOAD_DTYPE, order="C").ravel(order="C").tofile(f)
        np.asarray(mine_mask, dtype=PAYLOAD_DTYPE, order="C").ravel(order="C").tofile(f)


def read_snapshot_payload(path: Path, rows: int, cols: int) -> tuple[Array2D, Array2D, Array2D, Array2D]:
    expected_bytes = bytes_per_snapshot(rows, cols)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Snapshot payload size mismatch for {path}: expected {expected_bytes} bytes, got {actual_bytes} bytes"
        )

    raw = np.fromfile(path, dtype=PAYLOAD_DTYPE)
    planes = raw.reshape((len(PAYLOAD_PLANES), rows, cols))
    return planes[0], planes[1], planes[2], planes[3]


def load_index_rows(out_dir: Path) -> list[dict[str, str]]:
    index_path = out_dir / "index.csv"
    with index_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def sample_bin_files(out_dir: Path, n: int, seed: int) -> list[Path]:
    rows = load_index_rows(out_dir)

    if n <= 0 or not rows:
        return []

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [out_dir / rows[int(i)]["path"] for i in picks]


def confirm_proceed(num_samples: int, rows: int, cols: int) -> bool:
    sample_bytes = bytes_per_snapshot(rows, cols)
    estimated_bytes = sample_bytes * num_samples
    files_per_dir = files_per_dir_for_shape(rows, cols)
    n_dirs = (num_samples + files_per_dir - 1) // files_per_dir

    print("Dataset generation plan")
    print(f"difficulty={DIFFICULTY}")
    print(f"shape=({rows}, {cols})")
    print(f"samples={num_samples}")
    print(f"payload arrays={PAYLOAD_PLANES}")
    print(f"payload dtype={np.dtype(PAYLOAD_DTYPE).name}")
    print(f"bytes_per_snapshot={sample_bytes} ({format_bytes(sample_bytes)})")
    print(f"estimated total payload size={format_bytes(estimated_bytes)} ({estimated_bytes} bytes)")
    print(f"target payload per directory={format_bytes(TARGET_DIR_BYTES)} ({TARGET_DIR_BYTES} bytes)")
    print(f"computed files_per_dir={files_per_dir}")
    print(f"snapshot files={num_samples}")
    print(f"directories={n_dirs}")
    print(f"output_dir={OUT_DIR}")
    print()

    response = input("Proceed? [Y/n]: ").strip().lower()
    return response in ("", "y", "yes")


def generate_snapshot_dataset(
    difficulty: str,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
    out_dir: Path,
) -> GenerationSummary:
    rows = game_mode[difficulty]["rows"]
    cols = game_mode[difficulty]["columns"]
    files_per_dir = files_per_dir_for_shape(rows, cols)

    rng = np.random.default_rng(seed)
    sample_idx = 0
    game_idx = 0

    index_path = out_dir / "index.csv"
    game_summary_path = out_dir / "game_summary.csv"
    with (
        index_path.open("w", newline="", encoding="utf-8") as index_file,
        game_summary_path.open("w", newline="", encoding="utf-8") as game_file,
    ):
        writer = csv.writer(index_file)
        writer.writerow(
            [
                "sample_idx",
                "path",
                "game_idx",
                "game_seed",
                "step",
                "revealed_cells",
                "revealed_frac",
            ]
        )
        game_writer = csv.writer(game_file)
        game_writer.writerow(
            [
                "game_idx",
                "game_seed",
                "sample_count",
                "first_sample_idx",
                "last_sample_idx",
                "steps_simulated",
            ]
        )

        with tqdm(total=num_samples, desc="Writing snapshot bins", unit="sample", mininterval=0.2) as pbar:
            while sample_idx < num_samples:
                game_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                board = Minesweeper(difficulty=difficulty, seed=game_seed)
                reveal = board.random_safe_reveal if safe_reveal else board.random_reveal

                # Keep first reveal safe so each game contributes usable states.
                board.random_safe_reveal()

                kept_from_game = 0
                step = 0
                game_sample_count = 0
                first_sample_idx = -1
                last_sample_idx = -1

                while sample_idx < num_samples and not (board.game_over or board.game_won):
                    revealed = count_revealed(board)
                    if should_keep_sample(board, revealed, MAX_SAMPLES_PER_GAME, kept_from_game):
                        rel_path = sample_rel_path(sample_idx, files_per_dir)
                        abs_path = out_dir / rel_path

                        x_input, y_prob, covered_mask, mine_mask = snapshot_planes(board)
                        write_snapshot_payload(abs_path, x_input, y_prob, covered_mask, mine_mask)

                        revealed_frac = revealed / float(rows * cols)
                        writer.writerow(
                            [
                                sample_idx,
                                rel_path.as_posix(),
                                game_idx,
                                game_seed,
                                step,
                                revealed,
                                f"{revealed_frac:.6f}",
                            ]
                        )

                        if game_sample_count == 0:
                            first_sample_idx = sample_idx
                        last_sample_idx = sample_idx
                        game_sample_count += 1

                        sample_idx += 1
                        kept_from_game += 1
                        pbar.update(1)

                    reveal()
                    step += 1

                game_writer.writerow(
                    [
                        game_idx,
                        game_seed,
                        game_sample_count,
                        first_sample_idx,
                        last_sample_idx,
                        step,
                    ]
                )
                game_idx += 1

    return {
        "num_samples": sample_idx,
        "games_simulated": game_idx,
        "rows": rows,
        "cols": cols,
        "bytes_per_snapshot": bytes_per_snapshot(rows, cols),
        "files_per_dir": files_per_dir,
        "target_dir_bytes": TARGET_DIR_BYTES,
    }


def write_metadata(out_dir: Path, summary: GenerationSummary) -> None:
    metadata = {
        "difficulty": DIFFICULTY,
        "safe_reveal": SAFE_REVEAL,
        "seed": SEED,
        "num_samples": int(summary["num_samples"]),
        "games_simulated": int(summary["games_simulated"]),
        "rows": int(summary["rows"]),
        "cols": int(summary["cols"]),
        "payload": {
            "format": "raw_bin_no_header",
            "dtype": np.dtype(PAYLOAD_DTYPE).name,
            "planes": list(PAYLOAD_PLANES),
            "plane_count": len(PAYLOAD_PLANES),
            "bytes_per_snapshot": int(summary["bytes_per_snapshot"]),
        },
        "index_file": "index.csv",
        "game_summary_file": "game_summary.csv",
        "files_per_dir": int(summary["files_per_dir"]),
        "target_dir_bytes": int(summary["target_dir_bytes"]),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    rows = game_mode[DIFFICULTY]["rows"]
    cols = game_mode[DIFFICULTY]["columns"]

    if PROMPT_BEFORE_RUN and not confirm_proceed(NUM_SAMPLES, rows, cols):
        print("Aborted.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = generate_snapshot_dataset(
        difficulty=DIFFICULTY,
        num_samples=NUM_SAMPLES,
        safe_reveal=SAFE_REVEAL,
        seed=SEED,
        out_dir=OUT_DIR,
    )
    write_metadata(OUT_DIR, summary)

    print("Generated payload-only snapshot dataset")
    print(f"output_dir={OUT_DIR}")
    print(f"samples={summary['num_samples']}")
    print(f"games_simulated={summary['games_simulated']}")
    print(f"bytes_per_snapshot={summary['bytes_per_snapshot']}")
    print(f"files_per_dir={summary['files_per_dir']} (target={format_bytes(summary['target_dir_bytes'])})")
    print("layout per .bin payload: x_input, y_prob, covered_mask, mine_mask (all float64)")


if __name__ == "__main__":
    main()