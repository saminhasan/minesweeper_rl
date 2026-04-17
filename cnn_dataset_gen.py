from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from game_engine import Minesweeper, game_mode


# Binary layout: 4 planes packed sequentially per sample, all float64, row-major.
PAYLOAD_DTYPE = np.float64
PAYLOAD_PLANES = ("x_input", "y_prob", "covered_mask", "mine_mask")


def _snap(board: Minesweeper) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the four dataset planes from the current board state."""
    return (
        board.get_input().astype(PAYLOAD_DTYPE, copy=False),   # what the player sees
        board.get_output().astype(PAYLOAD_DTYPE, copy=False),  # Bayesian mine probabilities
        (board.state == board.states.COVERED).astype(PAYLOAD_DTYPE, copy=False),  # covered mask
        (board.mine_count == -1).astype(PAYLOAD_DTYPE, copy=False),               # true mine locations
    )


def generate(
    difficulty: str,
    num_samples: int,
    seed: int,
    out_dir: Path,
    max_per_game: int,
    min_frac: float,
    max_frac: float,
    safe_reveal: bool,
) -> None:
    cfg = game_mode[difficulty]
    rows, cols = cfg["rows"], cfg["columns"]
    total_cells = rows * cols
    n_planes = len(PAYLOAD_PLANES)
    snap_sz = n_planes * rows * cols * np.dtype(PAYLOAD_DTYPE).itemsize

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "data.bin"

    # Preallocate full binary file as memmap
    data = np.memmap(bin_path, dtype=PAYLOAD_DTYPE, mode="w+", shape=(num_samples, n_planes, rows, cols))

    rng = np.random.default_rng(seed)
    sample_idx = 0
    game_idx = 0

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as idx_f:
        writer = csv.writer(idx_f)
        writer.writerow(["sample_idx", "path", "offset_in_shard", "game_idx", "game_seed", "step", "revealed_cells"])

        try:
            with tqdm(total=num_samples, desc=f"[{difficulty}]", unit="sample") as pbar:
                while sample_idx < num_samples:
                    game_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                    board = Minesweeper(difficulty, seed=game_seed)
                    board.random_safe_reveal()  # first move is always safe
                    reveal = board.random_safe_reveal if safe_reveal else board.random_reveal
                    kept = 0
                    step = 0

                    while sample_idx < num_samples and not (board.game_over or board.game_won):
                        revealed = int(np.count_nonzero(board.state == board.states.UNCOVERED))
                        frac = revealed / total_cells
                        if (
                            kept < max_per_game
                            and min_frac <= frac <= max_frac
                            and np.any(board.get_frontier_cells())
                        ):
                            x, y, cov, mine = _snap(board)
                            data[sample_idx, 0] = x
                            data[sample_idx, 1] = y
                            data[sample_idx, 2] = cov
                            data[sample_idx, 3] = mine
                            writer.writerow([sample_idx, "data.bin", sample_idx, game_idx, game_seed, step, revealed])
                            sample_idx += 1
                            kept += 1
                            pbar.update(1)

                        reveal()
                        step += 1

                    game_idx += 1
        finally:
            del data  # flush memmap

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "difficulty": difficulty,
            "seed": seed,
            "num_samples": sample_idx,
            "games_simulated": game_idx,
            "rows": rows,
            "cols": cols,
            "payload": {
                "format": "raw_bin_no_header",
                "dtype": np.dtype(PAYLOAD_DTYPE).name,
                "planes": list(PAYLOAD_PLANES),
                "plane_count": n_planes,
                "bytes_per_snapshot": snap_sz,
                "samples_per_shard": num_samples,
            },
            "index_file": "index.csv",
        }, f, indent=2)

    print(f"Done: {sample_idx:,} samples, {game_idx:,} games -> {out_dir}")
    print(f"File: data.bin  ({snap_sz * num_samples / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    for level in ("easy", "intermediate", "hard"):
        LEVEL             = level  # "easy" | "intermediate" | "hard"
        SEED              = 123     # master RNG seed for reproducibility
        SAFE_REVEAL       = True    # use safe reveal for mid-game steps too
        NUM_SAMPLES       = 2**18   # fixed sample count for all levels
        MAX_PER_GAME      = 16      # max snapshots to keep from a single game
        MIN_REVEALED_FRAC = 0.01   # discard snapshots with < 1% of board revealed
        MAX_REVEALED_FRAC = 0.9999    # discard snapshots with > 99.99% of board revealed

        run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + f"_{LEVEL}_{NUM_SAMPLES}_{SEED}"
        out_dir = Path("data") / LEVEL / run_name

        generate(
            difficulty=LEVEL,
            num_samples=NUM_SAMPLES,
            seed=SEED,
            out_dir=out_dir,
            max_per_game=MAX_PER_GAME,
            min_frac=MIN_REVEALED_FRAC,
            max_frac=MAX_REVEALED_FRAC,
            safe_reveal=SAFE_REVEAL,
        )
