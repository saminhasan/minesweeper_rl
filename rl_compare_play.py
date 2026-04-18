from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import bayes_play
import rl_play
from predictor import Predictor


# -------------------------
# Config
# -------------------------
levels = ["easy", "intermediate", "hard"]

LEVEL      = levels[2]   # 0=easy, 1=intermediate, 2=hard
N_GAMES    = 200
SAFE_START = True
SEED       = 123

POLICY_MODEL_PATH = Path(f"models/RL/{LEVEL}/policy.keras")
OUT_DIR           = Path(f"logs/{LEVEL}/compare")


# -------------------------
# Helpers
# -------------------------
def print_gpu_status() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU available: no (running on CPU)")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"GPU available: yes ({len(gpus)} device(s))")
    for i, gpu in enumerate(gpus):
        print(f"  GPU[{i}]: {gpu.name}")


def save_comparison_plots(
    rows: list[dict[str, Any]],
    run_dir: Path,
    level: str,
    aggregate: dict[str, Any],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    n = len(rows)
    if n == 0:
        return

    games      = np.arange(1, n + 1)
    rl_won     = np.array([r["rl_won"]     for r in rows], dtype=np.float64)
    bayes_won  = np.array([r["bayes_won"]  for r in rows], dtype=np.float64)
    rl_steps   = np.array([r["rl_steps"]   for r in rows], dtype=np.float64)
    bayes_steps = np.array([r["bayes_steps"] for r in rows], dtype=np.float64)

    window = min(50, max(10, n // 10))

    def rolling(arr: np.ndarray) -> np.ndarray:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = max(0, i - window + 1)
            out[i] = np.mean(arr[s : i + 1])
        return out

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"RL vs Bayesian — {level.capitalize()}  ({n} games, rolling window={window})",
        fontsize=13,
    )

    # --- cumulative win % ---
    ax = axes[0, 0]
    cum_rl    = np.cumsum(rl_won)    / games * 100
    cum_bayes = np.cumsum(bayes_won) / games * 100
    ax.plot(games, cum_rl,    label="RL",       color="tab:blue",   linewidth=2)
    ax.plot(games, cum_bayes, label="Bayesian", color="tab:orange", linewidth=2)
    ax.axhline(aggregate["rl_win_rate"]    * 100, color="tab:blue",   linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(aggregate["bayes_win_rate"] * 100, color="tab:orange", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(n, cum_rl[-1]    + 1.5, f"{cum_rl[-1]:.1f}%",    ha="right", fontsize=9, color="tab:blue")
    ax.text(n, cum_bayes[-1] - 3.5, f"{cum_bayes[-1]:.1f}%", ha="right", fontsize=9, color="tab:orange")
    ax.set_ylabel("Win %  (cumulative)")
    ax.set_xlabel("Number of games")
    ax.set_title("Win % vs Games")
    ax.set_ylim(0, 108)
    ax.set_xlim(1, n)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- rolling avg steps ---
    ax = axes[0, 1]
    ax.plot(games, rolling(rl_steps),    label="RL",       color="tab:blue",   linewidth=2)
    ax.plot(games, rolling(bayes_steps), label="Bayesian", color="tab:orange", linewidth=2)
    ax.set_ylabel("Steps")
    ax.set_xlabel("Game")
    ax.set_title("Rolling avg steps per game")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- bar: final win rate ---
    ax = axes[1, 0]
    vals = [aggregate["rl_win_rate"], aggregate["bayes_win_rate"]]
    bars = ax.bar(["RL", "Bayesian"], vals, color=["tab:blue", "tab:orange"])
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Win rate")
    ax.set_title(f"Final win rate ({n} games)")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # --- bar: avg steps ---
    ax = axes[1, 1]
    vals_s = [aggregate["rl_avg_steps"], aggregate["bayes_avg_steps"]]
    bars_s = ax.bar(["RL", "Bayesian"], vals_s, color=["tab:blue", "tab:orange"])
    for bar, val in zip(bars_s, vals_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Avg steps")
    ax.set_title(f"Avg steps per game ({n} games)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = run_dir / "comparison_plots.png"
    fig.savefig(str(plot_path), dpi=160)
    plt.close(fig)
    print(f"Saved comparison plots: {plot_path}")


# -------------------------
# Main comparison runner
# -------------------------
def run_comparison(
    *,
    level: str,
    n_games: int,
    policy_model_path: Path,
    safe_start: bool,
) -> None:
    print_gpu_status()

    if not policy_model_path.exists():
        raise FileNotFoundError(f"Policy model not found: {policy_model_path}")

    policy = tf.keras.models.load_model(policy_model_path, compile=False)
    print(f"Loaded RL policy: {policy_model_path}")

    predictor = Predictor(level)

    # Suppress per-step logging inside sub-modules — we only need summaries
    bayes_play.LOG_STEPS = False
    rl_play.LOG_STEPS = False

    rng = np.random.default_rng(SEED)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_games)]

    rows: list[dict[str, Any]] = []

    print(f"\nRunning {n_games} games on level={level}  (identical seeds for both strategies)")
    print("-" * 66)

    with tqdm(range(n_games), desc="Compare", unit="game") as pbar:
        for ep in pbar:
            seed = seeds[ep]

            _, rl_summary = rl_play.play_one_game(
                episode_idx=ep,
                level=level,
                policy=policy,
                predictor=predictor,
                safe_start=safe_start,
                seed=seed,
            )

            _, bayes_summary = bayes_play.play_one_game(
                episode_idx=ep,
                level=level,
                safe_start=safe_start,
                seed=seed,
            )

            rows.append({
                "game":         ep + 1,
                "seed":         seed,
                "rl_won":       int(rl_summary["won"]),
                "rl_steps":     rl_summary["steps"],
                "rl_reward":    rl_summary["total_reward"],
                "bayes_won":    int(bayes_summary["won"]),
                "bayes_steps":  bayes_summary["steps"],
                "bayes_reward": bayes_summary["total_reward"],
            })

            rl_wr    = sum(r["rl_won"]    for r in rows) / len(rows)
            bayes_wr = sum(r["bayes_won"] for r in rows) / len(rows)
            pbar.set_postfix(
                rl_wr=f"{rl_wr:.3f}",
                b_wr=f"{bayes_wr:.3f}",
            )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = OUT_DIR / f"{timestamp}_{level}_{n_games}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["game", "seed", "rl_won", "rl_steps", "rl_reward",
                        "bayes_won", "bayes_steps", "bayes_reward"],
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, Any] = {
        "level":            level,
        "n_games":          n_games,
        "safe_start":       safe_start,
        "policy_model":     str(policy_model_path),
        "rl_win_rate":      sum(r["rl_won"]    for r in rows) / n_games,
        "bayes_win_rate":   sum(r["bayes_won"] for r in rows) / n_games,
        "rl_avg_steps":     float(np.mean([r["rl_steps"]    for r in rows])),
        "bayes_avg_steps":  float(np.mean([r["bayes_steps"] for r in rows])),
        "rl_avg_reward":    float(np.mean([r["rl_reward"]   for r in rows])),
        "bayes_avg_reward": float(np.mean([r["bayes_reward"] for r in rows])),
    }

    with (run_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    save_comparison_plots(rows, run_dir, level, aggregate)

    print("\nAggregate results:")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved to: {run_dir}")
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    run_comparison(
        level=LEVEL,
        n_games=N_GAMES,
        policy_model_path=POLICY_MODEL_PATH,
        safe_start=SAFE_START,
    )
