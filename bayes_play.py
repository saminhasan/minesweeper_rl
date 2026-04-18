from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from game_engine import Minesweeper, game_mode


# -------------------------
# Config
# -------------------------
levels = ["easy", "intermediate", "hard"]

LEVEL = levels[1]          # "easy" | "intermediate" | "hard"
N_GAMES = 100
SAFE_START = True
SEED = 123

LOG_DIR = Path(f"logs/{LEVEL}/bayes_play")
LOG_STEPS = True
LOG_SUMMARIES = True


# -------------------------
# Helpers
# -------------------------
def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_win_rate_plot(
    summary_rows: list[dict[str, Any]],
    run_dir: Path,
    level: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    n = len(summary_rows)
    if n == 0:
        return

    won = np.array([r["won"] for r in summary_rows], dtype=np.float64)
    games = np.arange(1, n + 1)
    cum_pct = np.cumsum(won) / games * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(games, cum_pct, linewidth=2, color="tab:blue")
    ax.axhline(cum_pct[-1], linestyle="--", color="tab:blue", alpha=0.5, linewidth=1)
    ax.text(n, cum_pct[-1] + 2, f"{cum_pct[-1]:.1f}%", ha="right", fontsize=11)
    ax.set_xlabel("Number of games")
    ax.set_ylabel("Win %  (cumulative)")
    ax.set_title(f"Bayesian — Win % vs Games  ({level.capitalize()},  n={n})")
    ax.set_xlim(1, n)
    ax.set_ylim(0, 108)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = run_dir / "win_rate.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Saved win rate plot: {plot_path}")


# -------------------------
# Board wrappers
# -------------------------
def new_board(level: str, seed: int | None = None) -> Minesweeper:
    return Minesweeper(level, seed=seed)


def board_random_safe_reveal(board: Minesweeper) -> None:
    board.random_safe_reveal()


def board_is_done(board: Minesweeper) -> bool:
    return bool(board.game_over or board.game_won)


def board_won(board: Minesweeper) -> bool:
    return bool(board.game_won)


def board_get_bayesian_prob(board: Minesweeper) -> np.ndarray:
    return np.asarray(board.get_output(), dtype=np.float32)


def board_covered_mask(board: Minesweeper) -> np.ndarray:
    return np.asarray(board.state == board.states.COVERED, dtype=np.float32)


def board_uncovered_count(board: Minesweeper) -> int:
    return int(board.total_cells - board.covered_count)


def board_frontier_count(board: Minesweeper) -> int:
    return int(np.count_nonzero(board.get_frontier_cells()))


def board_click(board: Minesweeper, row: int, col: int) -> None:
    board.reveal(row, col)


# -------------------------
# Bayesian action
# -------------------------
def bayesian_best_action(board: Minesweeper) -> tuple[int, int, np.ndarray]:
    mine_prob = board_get_bayesian_prob(board)
    covered_mask = board_covered_mask(board).astype(bool)

    # Only covered cells are valid.
    masked_prob = np.where(covered_mask, mine_prob, np.inf)

    # Break ties randomly among equal-min-risk cells.
    min_prob = float(np.min(masked_prob))
    candidates = np.flatnonzero(masked_prob.ravel() == min_prob)
    flat = int(np.random.choice(candidates))

    cols = board.n_cols
    row = flat // cols
    col = flat % cols
    return row, col, mine_prob


# -------------------------
# Reward
# -------------------------
REWARD_MINE_HIT = -1e3
REWARD_WIN_BONUS = 1e4
REWARD_FRONTIER_SCALE = 1.0


def compute_step_reward(
    uncovered_before: int,
    uncovered_after: int,
    frontier_before: int,
    frontier_after: int,
    hit_mine: bool,
    won: bool,
) -> float:
    if hit_mine:
        return float(REWARD_MINE_HIT)

    new_cells = max(0, uncovered_after - uncovered_before)
    reward = float(new_cells)

    frontier_reduction = max(0, frontier_before - frontier_after)
    reward += float(REWARD_FRONTIER_SCALE * frontier_reduction)

    if won:
        reward += float(REWARD_WIN_BONUS)

    return float(reward)


# -------------------------
# Play
# -------------------------
def play_one_game(
    *,
    episode_idx: int,
    level: str,
    safe_start: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    board = new_board(level, seed=seed)

    step_logs: list[dict[str, Any]] = []
    total_reward = 0.0
    steps = 0

    if safe_start and not board_is_done(board):
        board_random_safe_reveal(board)

    while not board_is_done(board):
        covered_mask = board_covered_mask(board)
        if float(np.sum(covered_mask)) <= 0.0:
            break

        row, col, mine_prob = bayesian_best_action(board)
        action = row * board.n_cols + col

        uncovered_before = board_uncovered_count(board)
        frontier_before = board_frontier_count(board)
        chosen_prob = float(mine_prob[row, col])

        board_click(board, row, col)

        uncovered_after = board_uncovered_count(board)
        frontier_after = board_frontier_count(board)
        hit_mine = bool(board.game_over)
        won = bool(board.game_won)

        reward = compute_step_reward(
            uncovered_before=uncovered_before,
            uncovered_after=uncovered_after,
            frontier_before=frontier_before,
            frontier_after=frontier_after,
            hit_mine=hit_mine,
            won=won,
        )
        total_reward += reward

        if LOG_STEPS:
            step_logs.append({
                "type": "step",
                "episode": episode_idx,
                "step": steps,
                "seed": seed,
                "action": action,
                "row": row,
                "col": col,
                "mine_prob": chosen_prob,
                "reward": reward,
                "uncovered_before": uncovered_before,
                "uncovered_after": uncovered_after,
                "frontier_before": frontier_before,
                "frontier_after": frontier_after,
                "game_over": hit_mine,
                "won": won,
            })

        steps += 1

    summary = {
        "type": "summary",
        "episode": episode_idx,
        "seed": seed,
        "level": level,
        "player": "bayesian",
        "won": bool(board_won(board)),
        "steps": steps,
        "total_reward": total_reward,
    }

    return step_logs, summary


def play_many_games(
    *,
    level: str,
    n_games: int,
    safe_start: bool,
) -> None:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = LOG_DIR / f"{timestamp}_{level}_{n_games}"
    run_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    wins = 0
    reward_sum = 0.0
    step_sum = 0

    rng = np.random.default_rng(SEED)

    with tqdm(range(n_games), desc="Bayes", unit="game") as pbar:
        for ep in pbar:
            seed = int(rng.integers(0, 2**31 - 1))
            ep_steps, ep_summary = play_one_game(
                episode_idx=ep,
                level=level,
                safe_start=safe_start,
                seed=seed,
            )

            if LOG_STEPS:
                step_rows.extend(ep_steps)
            if LOG_SUMMARIES:
                summary_rows.append(ep_summary)

            wins += int(ep_summary["won"])
            reward_sum += float(ep_summary["total_reward"])
            step_sum += int(ep_summary["steps"])

            pbar.set_postfix(
                won="T" if ep_summary["won"] else "F",
                wr=f"{wins / (ep + 1):.3f}",
                steps=ep_summary["steps"],
            )

    if LOG_STEPS:
        save_jsonl(run_dir / "steps.jsonl", step_rows)
    if LOG_SUMMARIES:
        save_jsonl(run_dir / "summaries.jsonl", summary_rows)

    aggregate = {
        "level": level,
        "n_games": n_games,
        "player": "bayesian",
        "safe_start": safe_start,
        "wins": wins,
        "win_rate": wins / n_games if n_games else 0.0,
        "avg_reward": reward_sum / n_games if n_games else 0.0,
        "avg_steps": step_sum / n_games if n_games else 0.0,
    }

    with (run_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    save_win_rate_plot(summary_rows, run_dir, level)

    print("\nDone.")
    print(json.dumps(aggregate, indent=2))
    print(f"Logs saved in: {run_dir}")


if __name__ == "__main__":
    play_many_games(
        level=LEVEL,
        n_games=N_GAMES,
        safe_start=SAFE_START,
    )