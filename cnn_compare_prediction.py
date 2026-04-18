from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from game_engine import Minesweeper
from predictor import Predictor


# -------------------------
# Config
# -------------------------
levels = ["easy", "intermediate", "hard"]

LEVEL          = levels[0]   # 0=easy, 1=intermediate, 2=hard
N_GAMES        = 100
SAFE_START     = True
SEED           = 123
PRED_THRESHOLD = 0.5         # probability cutoff for FP/FN classification

CNN_MODEL_PATH = Path(f"models/CNN/{LEVEL}/model.keras")
OUT_DIR        = Path(f"logs/{LEVEL}/compare_prediction")


# -------------------------
# Board helpers
# -------------------------
def new_board(level: str, seed: int | None = None) -> Minesweeper:
    return Minesweeper(level, seed=seed)


def board_is_done(board: Minesweeper) -> bool:
    return bool(board.game_over or board.game_won)


def get_covered_mask(board: Minesweeper) -> np.ndarray:
    return np.asarray(board.state == board.states.COVERED, dtype=bool)


def get_mine_truth(board: Minesweeper) -> np.ndarray:
    truth = np.zeros((board.n_rows, board.n_cols), dtype=np.float32)
    for r, c in board.mines:
        truth[r, c] = 1.0
    return truth


def bayesian_best_action(board: Minesweeper) -> tuple[int, int]:
    mine_prob = np.asarray(board.get_output(), dtype=np.float32)
    covered = board.state == board.states.COVERED
    masked = np.where(covered, mine_prob, np.inf)
    candidates = np.flatnonzero(masked.ravel() == float(np.min(masked)))
    flat = int(np.random.choice(candidates))
    return flat // board.n_cols, flat % board.n_cols


# -------------------------
# Metrics
# -------------------------
def brier_score(probs: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean((probs[mask] - truth[mask]) ** 2))


def false_rates(
    probs: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    threshold: float = PRED_THRESHOLD,
) -> tuple[float, float]:
    """Return (FPR, FNR) over masked cells at the given classification threshold.

    FPR = FP / (FP + TN)  — safe cells wrongly flagged as mines (over-caution)
    FNR = FN / (FN + TP)  — mine cells predicted safe (leads to clicking a mine → game loss)
    """
    if not mask.any():
        return float("nan"), float("nan")
    p = probs[mask]
    t = truth[mask]
    pred_pos   = p >= threshold
    actual_pos = t == 1.0
    tp = int(( pred_pos &  actual_pos).sum())
    fp = int(( pred_pos & ~actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos &  actual_pos).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return fpr, fnr


def build_calibration(
    all_probs: np.ndarray,
    all_truth: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    mean_pred = np.full(n_bins, np.nan)
    mean_true = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (all_probs >= lo) & (all_probs <= hi if i == n_bins - 1 else all_probs < hi)
        if mask.sum() > 0:
            mean_pred[i] = float(np.mean(all_probs[mask]))
            mean_true[i] = float(np.mean(all_truth[mask]))
            counts[i] = int(mask.sum())
    return centers, mean_pred, mean_true, counts


def build_roc(
    probs: np.ndarray,
    truth: np.ndarray,
    n_thresh: int = 300,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    pos = int((truth == 1).sum())
    neg = int((truth == 0).sum())
    if pos == 0 or neg == 0:
        return None
    thresholds = np.linspace(1.0, 0.0, n_thresh)
    tprs = np.empty(n_thresh)
    fprs = np.empty(n_thresh)
    for k, t in enumerate(thresholds):
        pp = probs >= t
        tprs[k] = float(((pp) & (truth == 1)).sum()) / pos
        fprs[k] = float(((pp) & (truth == 0)).sum()) / neg
    auc = float(np.trapezoid(tprs, fprs))
    if auc < 0.0:
        auc = -auc
    return fprs, tprs, auc


# -------------------------
# Plots
# -------------------------
def save_plots(
    rows: list[dict[str, Any]],
    bayes_probs_pool: np.ndarray,
    cnn_probs_pool: np.ndarray,
    truth_pool: np.ndarray,
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

    window = min(50, max(10, n // 10))

    def rolling(arr: np.ndarray) -> np.ndarray:
        out = np.empty(n)
        for i in range(n):
            s = max(0, i - window + 1)
            out[i] = np.nanmean(arr[s : i + 1])
        return out

    bayes_brier = np.array([r["bayes_brier"] for r in rows])
    cnn_brier   = np.array([r["cnn_brier"]   for r in rows])
    bayes_fnr   = np.array([r["bayes_fnr"]   for r in rows])
    cnn_fnr     = np.array([r["cnn_fnr"]     for r in rows])
    games       = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f"Bayesian vs CNN Prediction Quality — {level.capitalize()}  "
        f"({n} games, threshold={PRED_THRESHOLD}, window={window})",
        fontsize=13,
    )

    # --- [0,0] Calibration curves ---
    ax = axes[0, 0]
    _, bp, bt, _ = build_calibration(bayes_probs_pool, truth_pool)
    _, cp, ct, _ = build_calibration(cnn_probs_pool,   truth_pool)
    valid_b = ~np.isnan(bp) & ~np.isnan(bt)
    valid_c = ~np.isnan(cp) & ~np.isnan(ct)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect", alpha=0.5)
    ax.plot(bp[valid_b], bt[valid_b], "o-", color="tab:orange", label="Bayesian", linewidth=2, markersize=5)
    ax.plot(cp[valid_c], ct[valid_c], "s-", color="tab:blue",   label="CNN",      linewidth=2, markersize=5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual mine fraction")
    ax.set_title("Calibration (reliability diagram)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [0,1] ROC curves ---
    ax = axes[0, 1]
    roc_b = build_roc(bayes_probs_pool, truth_pool)
    roc_c = build_roc(cnn_probs_pool,   truth_pool)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    if roc_b is not None:
        bfprs, btprs, bauc = roc_b
        ax.plot(bfprs, btprs, color="tab:orange", linewidth=2, label=f"Bayesian (AUC={bauc:.3f})")
    if roc_c is not None:
        cfprs, ctprs, cauc = roc_c
        ax.plot(cfprs, ctprs, color="tab:blue",   linewidth=2, label=f"CNN      (AUC={cauc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (mine detection)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [0,2] FPR / FNR grouped bar chart (global, at threshold) ---
    ax = axes[0, 2]
    labels = [f"FPR\n(safe→flagged)", f"FNR\n(mine→missed)"]
    bayes_rates = [aggregate["bayes_fpr"], aggregate["bayes_fnr"]]
    cnn_rates   = [aggregate["cnn_fpr"],   aggregate["cnn_fnr"]]
    x     = np.arange(len(labels))
    width = 0.35
    bars_b = ax.bar(x - width / 2, bayes_rates, width, label="Bayesian", color="tab:orange")
    bars_c = ax.bar(x + width / 2, cnn_rates,   width, label="CNN",      color="tab:blue")
    for bar, val in list(zip(bars_b, bayes_rates)) + list(zip(bars_c, cnn_rates)):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title(f"False positive / negative rates\n(threshold={PRED_THRESHOLD}, pooled over all steps)")
    ax.set_ylim(0, min(1.15, max(max(bayes_rates), max(cnn_rates)) * 1.4 + 0.1))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- [1,0] Rolling Brier score per game ---
    ax = axes[1, 0]
    ax.plot(games, rolling(bayes_brier), color="tab:orange", linewidth=2, label="Bayesian")
    ax.plot(games, rolling(cnn_brier),   color="tab:blue",   linewidth=2, label="CNN")
    ax.axhline(aggregate["bayes_avg_brier"], color="tab:orange", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(aggregate["cnn_avg_brier"],   color="tab:blue",   linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Game")
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title(f"Rolling avg Brier score (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [1,1] Rolling FNR per game (most dangerous metric) ---
    ax = axes[1, 1]
    ax.plot(games, rolling(bayes_fnr), color="tab:orange", linewidth=2, label="Bayesian")
    ax.plot(games, rolling(cnn_fnr),   color="tab:blue",   linewidth=2, label="CNN")
    ax.axhline(aggregate["bayes_fnr"], color="tab:orange", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(aggregate["cnn_fnr"],   color="tab:blue",   linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Game")
    ax.set_ylabel("False negative rate (lower = safer)")
    ax.set_title(f"Rolling avg FNR — mines predicted safe (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [1,2] Summary bar: Brier + AUC ---
    ax = axes[1, 2]
    metrics = ["Avg Brier\n(lower=better)", "AUC-ROC\n(higher=better)"]
    bayes_v = [aggregate["bayes_avg_brier"], aggregate.get("bayes_auc", 0.0)]
    cnn_v   = [aggregate["cnn_avg_brier"],   aggregate.get("cnn_auc",   0.0)]
    x2      = np.arange(len(metrics))
    bars_b2 = ax.bar(x2 - width / 2, bayes_v, width, label="Bayesian", color="tab:orange")
    bars_c2 = ax.bar(x2 + width / 2, cnn_v,   width, label="CNN",      color="tab:blue")
    for bar, val in list(zip(bars_b2, bayes_v)) + list(zip(bars_c2, cnn_v)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics)
    ax.set_title(f"Overall metrics ({n} games)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = run_dir / "prediction_comparison_plots.png"
    fig.savefig(str(plot_path), dpi=160)
    plt.close(fig)
    print(f"Saved plots: {plot_path}")


# -------------------------
# Main runner
# -------------------------
def run_comparison(
    *,
    level: str,
    n_games: int,
    cnn_model_path: Path,
    safe_start: bool,
) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"GPU: {'yes (' + str(len(gpus)) + ' device(s))' if gpus else 'no (CPU)'}")

    if not cnn_model_path.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_model_path}")

    cnn_pred = Predictor(level, source="cnn")

    rng = np.random.default_rng(SEED)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_games)]

    rows: list[dict[str, Any]] = []

    # Pools for global calibration / ROC (covered cells only, all steps, all games)
    bayes_pool: list[np.ndarray] = []
    cnn_pool:   list[np.ndarray] = []
    truth_pool: list[np.ndarray] = []

    print(f"\nRunning {n_games} games on level={level}  (Bayesian trajectory, predictions compared at each step)")
    print(f"Classification threshold for FP/FN: {PRED_THRESHOLD}")
    print("-" * 70)

    with tqdm(range(n_games), desc="Predict", unit="game") as pbar:
        for ep in pbar:
            seed = seeds[ep]
            board = new_board(level, seed=seed)

            if safe_start and not board_is_done(board):
                board.random_safe_reveal()

            game_bayes_briers: list[float] = []
            game_cnn_briers:   list[float] = []
            game_bayes_fprs:   list[float] = []
            game_bayes_fnrs:   list[float] = []
            game_cnn_fprs:     list[float] = []
            game_cnn_fnrs:     list[float] = []
            n_predictions = 0
            steps = 0

            while not board_is_done(board):
                covered = get_covered_mask(board)
                if not covered.any():
                    break

                truth         = get_mine_truth(board)
                bayes_prob    = np.asarray(board.get_output(), dtype=np.float32)
                cnn_mine_prob, _ = cnn_pred.predict(board)

                game_bayes_briers.append(brier_score(bayes_prob,    truth, covered))
                game_cnn_briers.append(  brier_score(cnn_mine_prob, truth, covered))

                fpr_b, fnr_b = false_rates(bayes_prob,    truth, covered)
                fpr_c, fnr_c = false_rates(cnn_mine_prob, truth, covered)
                game_bayes_fprs.append(fpr_b)
                game_bayes_fnrs.append(fnr_b)
                game_cnn_fprs.append(fpr_c)
                game_cnn_fnrs.append(fnr_c)

                bayes_pool.append(bayes_prob[covered])
                cnn_pool.append(cnn_mine_prob[covered])
                truth_pool.append(truth[covered])
                n_predictions += int(covered.sum())

                row, col = bayesian_best_action(board)
                board.reveal(row, col)
                steps += 1

            def _avg(lst: list[float]) -> float:
                return float(np.nanmean(lst)) if lst else float("nan")

            game_bayes_brier = _avg(game_bayes_briers)
            game_cnn_brier   = _avg(game_cnn_briers)
            game_bayes_fpr   = _avg(game_bayes_fprs)
            game_bayes_fnr   = _avg(game_bayes_fnrs)
            game_cnn_fpr     = _avg(game_cnn_fprs)
            game_cnn_fnr     = _avg(game_cnn_fnrs)

            rows.append({
                "game":          ep + 1,
                "seed":          seed,
                "won":           int(board.game_won),
                "steps":         steps,
                "n_predictions": n_predictions,
                "bayes_brier":   game_bayes_brier,
                "cnn_brier":     game_cnn_brier,
                "bayes_fpr":     game_bayes_fpr,
                "bayes_fnr":     game_bayes_fnr,
                "cnn_fpr":       game_cnn_fpr,
                "cnn_fnr":       game_cnn_fnr,
            })

            pbar.set_postfix(
                b_brier=f"{game_bayes_brier:.4f}",
                c_brier=f"{game_cnn_brier:.4f}",
                b_fnr=f"{game_bayes_fnr:.3f}",
                c_fnr=f"{game_cnn_fnr:.3f}",
            )

    # --- Pool arrays ---
    all_bayes = np.concatenate(bayes_pool)
    all_cnn   = np.concatenate(cnn_pool)
    all_truth = np.concatenate(truth_pool)
    all_mask  = np.ones(all_bayes.size, dtype=bool)

    roc_b = build_roc(all_bayes, all_truth)
    roc_c = build_roc(all_cnn,   all_truth)

    global_bayes_fpr, global_bayes_fnr = false_rates(all_bayes, all_truth, all_mask)
    global_cnn_fpr,   global_cnn_fnr   = false_rates(all_cnn,   all_truth, all_mask)

    # --- Save outputs ---
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = OUT_DIR / f"{timestamp}_{level}_{n_games}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "prediction_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game", "seed", "won", "steps", "n_predictions",
                "bayes_brier", "cnn_brier",
                "bayes_fpr", "bayes_fnr",
                "cnn_fpr",   "cnn_fnr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, Any] = {
        "level":              level,
        "n_games":            n_games,
        "safe_start":         safe_start,
        "cnn_model":          str(cnn_model_path),
        "pred_threshold":     PRED_THRESHOLD,
        "total_predictions":  int(all_bayes.size),
        "bayes_avg_brier":    float(np.nanmean([r["bayes_brier"] for r in rows])),
        "cnn_avg_brier":      float(np.nanmean([r["cnn_brier"]   for r in rows])),
        "bayes_auc":          roc_b[2] if roc_b is not None else float("nan"),
        "cnn_auc":            roc_c[2] if roc_c is not None else float("nan"),
        "bayes_fpr":          global_bayes_fpr,
        "bayes_fnr":          global_bayes_fnr,
        "cnn_fpr":            global_cnn_fpr,
        "cnn_fnr":            global_cnn_fnr,
    }

    with (run_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    save_plots(rows, all_bayes, all_cnn, all_truth, run_dir, level, aggregate)

    print("\nAggregate results:")
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved to: {run_dir}")
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    run_comparison(
        level=LEVEL,
        n_games=N_GAMES,
        cnn_model_path=CNN_MODEL_PATH,
        safe_start=SAFE_START,
    )
