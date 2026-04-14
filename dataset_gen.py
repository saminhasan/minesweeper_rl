from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from game_engine import Minesweeper, game_mode

DIFFICULTY = "xtreme"
NUM_SAMPLES = 2**20
SAFE_REVEAL = False
SEED = 123
OUT = f"data/{DIFFICULTY}/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_minesweeper_{DIFFICULTY}_{NUM_SAMPLES}_{SEED}.npz"
SAVE_DATASET = False

MAX_SAMPLES_PER_GAME = 8
MIN_REVEALED_FRAC = 0.05
MAX_REVEALED_FRAC = 0.95


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

    frontier_count = int(np.count_nonzero(board.get_frontier_cells()))
    if frontier_count == 0:
        return False

    return True


def generate_dataset(
    difficulty: str,
    num_samples: int,
    safe_reveal: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)

    rows = game_mode[difficulty]["rows"]
    cols = game_mode[difficulty]["columns"]
    covered = float(Minesweeper(difficulty=difficulty, seed=0).states.COVERED)

    x_arr = np.empty((num_samples, rows, cols), dtype=np.float32)
    y_arr = np.empty((num_samples, rows, cols), dtype=np.float32)
    mask_arr = np.empty((num_samples, rows, cols), dtype=np.float32)

    sample_idx = 0
    games_played = 0

    with tqdm(total=num_samples, desc="samples", unit="sample") as pbar:
        while sample_idx < num_samples:
            board = Minesweeper(
                difficulty=difficulty,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            games_played += 1

            reveal = board.random_safe_reveal if safe_reveal else board.random_reveal
            board.random_safe_reveal()

            kept_from_game = 0
            step = 0

            while sample_idx < num_samples and not (board.game_over or board.game_won):
                revealed = count_revealed(board)

                if should_keep_sample(board, revealed, MAX_SAMPLES_PER_GAME, kept_from_game):
                    x_arr[sample_idx] = board.get_input()
                    y_arr[sample_idx] = board.get_output()
                    mask_arr[sample_idx] = x_arr[sample_idx] == covered

                    # Debug only:
                    # print(f"sample={sample_idx} game={games_played} step={step} revealed={revealed}")

                    sample_idx += 1
                    kept_from_game += 1
                    pbar.update(1)

                reveal()
                step += 1

            if games_played % 1000 == 0:
                pbar.set_postfix(games=games_played)

    return x_arr, y_arr, mask_arr


def main() -> None:
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_arr, y_arr, mask_arr = generate_dataset(
        difficulty=DIFFICULTY,
        num_samples=NUM_SAMPLES,
        safe_reveal=SAFE_REVEAL,
        seed=SEED,
    )

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