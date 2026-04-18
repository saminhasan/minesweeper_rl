from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from game_engine import Minesweeper, game_mode
from predictor import Predictor
from rl import build_or_load_policy, LEVEL_POLICY_CONFIGS

# -------------------------
# Hyperparameters
# -------------------------
SEED                         = 123
LEARNING_RATE                = 1e-3
VALUE_COEF                   = 0.1
GRAD_CLIP_NORM               = 1.0

LOG_EVERY_WINS               = 16
SAVE_EVERY_WINS              = 5
MIN_IMPROVEMENT              = 1e-12

MODEL_FILE                   = "policy.keras"
BEST_METRICS_FILE            = "pretrain_best_metrics.json"
LAST_RUN_METRICS_FILE        = "pretrain_last_run_metrics.json"

# supervised pretraining settings
PRETRAIN_EPOCHS_PER_UPDATE   = 8
PRETRAIN_BATCH_SIZE          = 128
UPDATE_AFTER_WINS            = 16

# discounted return for value-head warm start
GAMMA                        = 0.99

# keep same cheat-style opening behavior as train_rl
SAFE_START                   = True

# -------------------------
# Generic helpers
# -------------------------
def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


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


# -------------------------
# Board adapter layer
# -------------------------
def new_board(level: str, seed: int | None = None) -> Minesweeper:
    return Minesweeper(level, seed=seed)


def board_random_safe_reveal(board: Minesweeper) -> None:
    board.random_safe_reveal()


def board_is_done(board: Minesweeper) -> bool:
    return bool(board.game_over or board.game_won)


def board_won(board: Minesweeper) -> bool:
    return bool(board.game_won)


def board_get_input(board: Minesweeper) -> np.ndarray:
    return np.asarray(board.get_input(), dtype=np.float32)


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
# Policy utilities
# -------------------------
def make_action_mask(covered_mask: np.ndarray) -> np.ndarray:
    return covered_mask.reshape(-1).astype(np.float32, copy=False)


def masked_logits(logits: tf.Tensor, action_mask: tf.Tensor) -> tf.Tensor:
    big_neg = tf.constant(-1e9, dtype=logits.dtype)
    return tf.where(action_mask > 0.5, logits, big_neg)


def action_to_row_col(action: int, cols: int) -> tuple[int, int]:
    return int(action // cols), int(action % cols)


def teacher_action_from_bayes_safe(board: Minesweeper) -> int:
    mine_prob = board_get_bayesian_prob(board)
    safe_covered = (board.state == board.states.COVERED) & (board.mine_count != -1)
    masked = np.where(safe_covered, mine_prob, np.inf)

    flat = int(np.argmin(masked))
    if not np.isfinite(masked.reshape(-1)[flat]):
        raise ValueError("No safe covered action found")
    return flat


def discounted_returns_into(
    rewards: np.ndarray,
    returns_out: np.ndarray,
    steps: int,
    gamma: float,
) -> None:
    running = 0.0
    for i in range(steps - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        returns_out[i] = running


# -------------------------
# Reward shaping
# same style as train_rl
# -------------------------
def compute_step_reward(
    uncovered_before: int,
    uncovered_after: int,
    frontier_before: int,
    frontier_after: int,
    hit_mine: bool,
    won: bool,
    total_cells_for_level: int = 1,
) -> float:
    if hit_mine:
        return -1.0

    new_cells = max(0, uncovered_after - uncovered_before)
    reward = float(new_cells) / total_cells_for_level

    frontier_reduction = max(0, frontier_before - frontier_after)
    reward += 0.5 * float(frontier_reduction) / total_cells_for_level

    if won:
        reward += 1.0

    return float(reward)



# -------------------------
# Collect ONE teacher episode into preallocated scratch buffers
# Oracle-safe => should almost always win
# -------------------------
def collect_teacher_episode_into(
    *,
    level: str,
    predictor: Predictor,
    safe_start: bool,
    difficulty_seed: int | None,
    states_out: np.ndarray,
    masks_out: np.ndarray,
    actions_out: np.ndarray,
    rewards_out: np.ndarray,
) -> tuple[int, bool]:
    board = new_board(level, seed=difficulty_seed)

    # make sure mines exist before oracle-safe teacher acts
    if safe_start or not getattr(board, "_mines_placed", False):
        board_random_safe_reveal(board)

    cfg = game_mode[level]
    cols = int(cfg["columns"])
    total_safe_cells = (cfg["rows"] * cfg["columns"]) - cfg["mines"]

    steps = 0

    while not board_is_done(board):
        state, action_mask = predictor.build_state(board)

        if float(np.sum(action_mask)) <= 0.0:
            return steps, False

        action = teacher_action_from_bayes_safe(board)
        row, col = action_to_row_col(action, cols)

        uncovered_before = board_uncovered_count(board)
        frontier_before = board_frontier_count(board)

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
            total_cells_for_level=total_safe_cells,
        )

        states_out[steps] = state
        masks_out[steps] = action_mask
        actions_out[steps] = action
        rewards_out[steps] = reward
        steps += 1

        if hit_mine:
            return steps, False

    return steps, bool(board_won(board))


# -------------------------
# Supervised teacher update
# hard imitation + value warm start
# -------------------------
def supervised_teacher_update(
    policy: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    states: np.ndarray,
    action_masks: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
) -> tuple[float, float, float]:
    n = states.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    indices = np.arange(n)
    last_total = float("nan")
    last_policy = float("nan")
    acc_sum = 0.0
    acc_count = 0

    for _ in range(PRETRAIN_EPOCHS_PER_UPDATE):
        np.random.shuffle(indices)

        for start in range(0, n, PRETRAIN_BATCH_SIZE):
            batch_idx = indices[start:start + PRETRAIN_BATCH_SIZE]

            states_t = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
            masks_t = tf.convert_to_tensor(action_masks[batch_idx], dtype=tf.float32)
            actions_t = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
            returns_t = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits, values = policy(states_t, training=True)
                values = tf.squeeze(values, axis=-1)

                masked = masked_logits(logits, masks_t)

                policy_loss_vec = tf.keras.losses.sparse_categorical_crossentropy(
                    actions_t, masked, from_logits=True
                )
                policy_loss = tf.reduce_mean(policy_loss_vec)

                value_loss = tf.reduce_mean(tf.square(returns_t - values))

                total_loss = policy_loss + VALUE_COEF * value_loss

            grads = tape.gradient(total_loss, policy.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            optimizer.apply_gradients(zip(grads, policy.trainable_variables))

            pred = tf.argmax(masked, axis=-1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(pred, actions_t), tf.float32))

            last_total = float(total_loss.numpy())
            last_policy = float(policy_loss.numpy())
            acc_sum += float(acc.numpy()) * len(batch_idx)
            acc_count += len(batch_idx)

    mean_acc = acc_sum / max(1, acc_count)
    return last_total, last_policy, mean_acc


# -------------------------
# Training plots
# -------------------------
def save_training_plots(
    loss_hist: list[float],
    acc_hist: list[float],
    wins_hist: list[float],
    model_dir: Path,
    level: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping training plots.")
        return

    n = len(loss_hist)
    if n == 0:
        return

    x = np.arange(1, n + 1)
    window = min(50, max(5, n // 20))

    def rolling(data: list[float]) -> np.ndarray:
        arr = np.array(data, dtype=np.float64)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s = max(0, i - window + 1)
            out[i] = np.mean(arr[s:i + 1])
        return out

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Teacher Pretraining — {level} (rolling window={window})", fontsize=13)

    axes[0].plot(x, loss_hist, alpha=0.15, linewidth=0.6)
    axes[0].plot(x, rolling(loss_hist), linewidth=1.8)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, acc_hist, alpha=0.15, linewidth=0.6)
    axes[1].plot(x, rolling(acc_hist), linewidth=1.8)
    axes[1].set_ylabel("Teacher acc")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)



    plt.tight_layout()
    out_path = model_dir / "pretrain_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved pretraining curves: {out_path}")


# -------------------------
# Pretraining
# -------------------------
def pretrain(
    level: str,
    model_dir: Path,
    n_games: int,
    base_channels: int,
    conv_layers: int,
    body_dense_layers: int,
    head_dense_layers: int,
    model_file: str = MODEL_FILE,
    safe_start: bool = True,
) -> dict[str, Any]:
    set_global_seed(SEED)
    print_gpu_status()

    predictor = Predictor(level)

    cfg = game_mode[level]
    rows = int(cfg["rows"])
    cols = int(cfg["columns"])
    in_channels = 6
    total_safe_cells = (rows * cols) - int(cfg["mines"])

    model_path = model_dir / model_file
    best_metrics_path = model_dir / BEST_METRICS_FILE
    last_metrics_path = model_dir / LAST_RUN_METRICS_FILE

    policy, architecture_mismatch = build_or_load_policy(
        rows=rows,
        cols=cols,
        in_channels=in_channels,
        model_path=model_path,
        base_channels=base_channels,
        conv_layers=conv_layers,
        body_dense_layers=body_dense_layers,
        head_dense_layers=head_dense_layers,
        level_name=level,
    )

    if architecture_mismatch and model_path.exists():
        try:
            model_path.unlink()
            print(f"Removed stale file: {model_path}")
        except OSError as exc:
            print(f"Could not remove {model_path} ({exc})")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    previous_best = load_json(best_metrics_path)
    best_acc = float(previous_best["best_teacher_acc"]) if previous_best else -1.0

    loss_hist: list[float] = []
    acc_hist: list[float] = []
    wins_hist: list[float] = []

    # preallocate per-episode scratch
    ep_states = np.empty((total_safe_cells, rows, cols, in_channels), dtype=np.float32)
    ep_masks = np.empty((total_safe_cells, rows * cols), dtype=np.float32)
    ep_actions = np.empty((total_safe_cells,), dtype=np.int32)
    ep_rewards = np.empty((total_safe_cells,), dtype=np.float32)
    ep_returns = np.empty((total_safe_cells,), dtype=np.float32)

    # preallocate per-update batch buffers
    batch_capacity = UPDATE_AFTER_WINS * total_safe_cells
    batch_states = np.empty((batch_capacity, rows, cols, in_channels), dtype=np.float32)
    batch_masks = np.empty((batch_capacity, rows * cols), dtype=np.float32)
    batch_actions = np.empty((batch_capacity,), dtype=np.int32)
    batch_returns = np.empty((batch_capacity,), dtype=np.float32)

    batch_write = 0
    games_in_batch = 0
    games = 0

    print(
        f"level={level}, board={rows}x{cols}, n_games={n_games}, "
    )

    with tqdm(total=n_games, desc="Pretrain", unit="game") as pbar:
        while games < n_games:
            steps, _ = collect_teacher_episode_into(
                level=level,
                predictor=predictor,
                safe_start=safe_start,
                difficulty_seed=int(np.random.randint(0, 2**31 - 1)),
                states_out=ep_states,
                masks_out=ep_masks,
                actions_out=ep_actions,
                rewards_out=ep_rewards,
            )

            discounted_returns_into(ep_rewards, ep_returns, steps, GAMMA)

            batch_states[batch_write:batch_write + steps] = ep_states[:steps]
            batch_masks[batch_write:batch_write + steps] = ep_masks[:steps]
            batch_actions[batch_write:batch_write + steps] = ep_actions[:steps]
            batch_returns[batch_write:batch_write + steps] = ep_returns[:steps]

            batch_write += steps
            games_in_batch += 1
            games += 1
            pbar.update(1)
            pbar.set_postfix(steps=steps)

            should_update = (games_in_batch >= UPDATE_AFTER_WINS) or (games == n_games)
            if not should_update:
                continue

            total_loss, policy_loss, teacher_acc = supervised_teacher_update(
                policy=policy,
                optimizer=optimizer,
                states=batch_states[:batch_write],
                action_masks=batch_masks[:batch_write],
                actions=batch_actions[:batch_write],
                returns=batch_returns[:batch_write],
            )

            loss_hist.append(total_loss)
            acc_hist.append(teacher_acc)
            wins_hist.append(float(games_in_batch))

            batch_write = 0
            games_in_batch = 0
            pbar.set_postfix(loss=f"{total_loss:.4f}", acc=f"{teacher_acc:.4f}", steps=steps)

            now = datetime.now().isoformat(timespec="seconds")
            run_metrics = {
                "timestamp": now,
                "level": level,
                "model_dir": str(model_dir),
                "model_file": model_file,
                "architecture_mismatch": bool(architecture_mismatch),
                "base_channels": base_channels,
                "n_games": n_games,
                "games_collected": games,
                "teacher_total_loss": float(total_loss),
                "teacher_policy_loss": float(policy_loss),
                "teacher_acc": float(teacher_acc),
                "best_teacher_acc": float(best_acc),
            }
            save_json(last_metrics_path, run_metrics)

            if games % SAVE_EVERY_WINS == 0 or games == n_games:
                improved = teacher_acc > best_acc + MIN_IMPROVEMENT
                if improved:
                    best_acc = teacher_acc
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    policy.save(model_path)
                    save_json(best_metrics_path, {**run_metrics, "model_path": str(model_path), "best_teacher_acc": best_acc})
                    print(f"  >> Pretrain improved: acc={teacher_acc:.4f} saved -> {model_path}")
                elif games == n_games:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    policy.save(model_path)
                    print(f"  >> Final pretrained policy saved -> {model_path}")

    save_training_plots(loss_hist, acc_hist, wins_hist, model_dir, level)

    final_metrics = load_json(last_metrics_path)
    if final_metrics is None:
        final_metrics = {
            "level": level,
            "games_collected": games,
            "best_teacher_acc": best_acc,
        }
    return final_metrics


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    levels = ["easy", "intermediate", "hard"]
    ACTIVE_LEVEL = levels[2]          # ← change index: 0=easy, 1=intermediate, 2=hard

    LEVEL_CONFIGS: dict[str, dict[str, Any]] = {
        "easy": {
            "model_dir": Path("models/RL/easy"),
            "model_file": "policy.keras",
            "n_games": 2048,
            "safe_start": True,
        },
        "intermediate": {
            "model_dir": Path("models/RL/intermediate"),
            "model_file": "policy.keras",
            "n_games": 2048,
            "safe_start": True,
        },
        "hard": {
            "model_dir": Path("models/RL/hard"),
            "model_file": "policy.keras",
            "n_games": 2048,
            "safe_start": True,
        },
    }

    cfg = LEVEL_CONFIGS[ACTIVE_LEVEL]
    arch = LEVEL_POLICY_CONFIGS[ACTIVE_LEVEL]

    pretrain(
        level=ACTIVE_LEVEL,
        model_dir=cfg["model_dir"],
        n_games=cfg["n_games"],
        model_file=cfg["model_file"],
        base_channels=arch["base_channels"],
        conv_layers=arch["conv_layers"],
        body_dense_layers=arch["body_dense_layers"],
        head_dense_layers=arch["head_dense_layers"],
        safe_start=cfg["safe_start"],
    )