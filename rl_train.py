from __future__ import annotations

import csv
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


levels = ["easy", "intermediate", "hard"]
ACTIVE_LEVEL = levels[1]          # "easy" | "intermediate" | "hard"
# -------------------------
# Hyperparameters
# -------------------------
SEED                         = 123
GAMMA                        = 0.99
GAE_LAMBDA                   = 0.95
LEARNING_RATE                = 1e-4
ENTROPY_BETA                 = 1e-2
VALUE_COEF                   = 0.1
GRAD_CLIP_NORM               = 0.5

PPO_CLIP_EPS                 = 0.2
PPO_EPOCHS                   = 8
PPO_BATCH_SIZE               = 64

REWARD_MINE_HIT              = -1.0
REWARD_WIN_BONUS             = 1.0
REWARD_FRONTIER_SCALE        = 0.5

LOG_EVERY_EPISODES           = 10
SAVE_EVERY_EPISODES          = 5
WIN_RATE_WINDOW              = 5
MIN_IMPROVEMENT              = 1e-12

MODEL_FILE                   = "policy.keras"
BEST_METRICS_FILE            = "best_metrics.json"
LAST_RUN_METRICS_FILE        = "last_run_metrics.json"
CURVE_ARTIFACTS_DIR          = "training_curves"

# Optional evaluation behavior while collecting actions
GREEDY_ACTION_PROB           = 0.5

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


def sample_action(logits: tf.Tensor, action_mask: tf.Tensor) -> tuple[int, tf.Tensor, tf.Tensor]:
    masked = masked_logits(logits, action_mask)
    dist = tf.random.categorical(masked[None, :], num_samples=1)
    action = int(dist[0, 0].numpy())
    log_probs = tf.nn.log_softmax(masked)
    probs = tf.nn.softmax(masked)
    log_prob = log_probs[action]
    entropy = -tf.reduce_sum(probs * log_probs)
    return action, log_prob, entropy


def greedy_action(logits: tf.Tensor, action_mask: tf.Tensor) -> int:
    masked = masked_logits(logits, action_mask)
    return int(tf.argmax(masked).numpy())


def compute_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in range(n - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:n]
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize_array(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    return ((x - mu) / (sigma + 1e-8)).astype(np.float32)


# -------------------------
# Reward shaping
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
        return float(REWARD_MINE_HIT)

    new_cells = max(0, uncovered_after - uncovered_before)
    reward = float(new_cells) / total_cells_for_level

    frontier_reduction = max(0, frontier_before - frontier_after)
    reward += float(REWARD_FRONTIER_SCALE * frontier_reduction) / total_cells_for_level

    if won:
        reward += float(REWARD_WIN_BONUS)

    return float(reward)



def action_to_row_col(action: int, cols: int) -> tuple[int, int]:
    return int(action // cols), int(action % cols)


# -------------------------
# Episode rollout
# Pure RL collection, no teacher
# -------------------------
def play_episode(
    *,
    level: str,
    policy: tf.keras.Model,
    predictor: Predictor,
    safe_start: bool,
    difficulty_seed: int | None,
) -> dict[str, Any]:
    board = new_board(level, seed=difficulty_seed)
    if safe_start:
        board_random_safe_reveal(board)

    cfg = game_mode[level]
    cols = int(cfg["columns"])
    total_safe_cells = (cfg["rows"] * cfg["columns"]) - cfg["mines"]

    states: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    rewards: list[float] = []

    total_reward = 0.0
    steps = 0
    entropy_accum = 0.0

    while not board_is_done(board):
        state, action_mask = predictor.build_state(board)

        if float(np.sum(action_mask)) <= 0.0:
            break

        state_t = tf.convert_to_tensor(state[None, ...], dtype=tf.float32)
        logits_t, value_t = policy(state_t, training=False)
        logits = logits_t[0]
        value = float(value_t[0, 0].numpy())

        if GREEDY_ACTION_PROB > 0.0 and np.random.rand() < GREEDY_ACTION_PROB:
            action = greedy_action(logits, tf.convert_to_tensor(action_mask, dtype=tf.float32))
            masked = masked_logits(logits, tf.convert_to_tensor(action_mask, dtype=tf.float32))
            log_prob = tf.nn.log_softmax(masked)[action]
            probs = tf.nn.softmax(masked)
            entropy = -tf.reduce_sum(probs * tf.nn.log_softmax(masked))
        else:
            action, log_prob, entropy = sample_action(
                logits, tf.convert_to_tensor(action_mask, dtype=tf.float32)
            )

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

        states.append(state)
        action_masks.append(action_mask)
        actions.append(action)
        old_log_probs.append(float(log_prob.numpy()))
        old_values.append(value)
        rewards.append(float(reward))

        total_reward += float(reward)
        entropy_accum += float(entropy.numpy())
        steps += 1

    return {
        "states": np.asarray(states, dtype=np.float32),
        "action_masks": np.asarray(action_masks, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "old_values": np.asarray(old_values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "reward": float(total_reward),
        "steps": int(steps),
        "won": bool(board_won(board)),
        "entropy": float(entropy_accum / max(1, steps)),
    }


# -------------------------
# PPO update
# -------------------------
def ppo_update(
    policy: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    rollout: dict[str, Any],
) -> float:
    actions = rollout["actions"]
    if actions.size == 0:
        return float("nan")

    states = rollout["states"]
    masks = rollout["action_masks"]
    old_log_probs = rollout["old_log_probs"]
    old_values = rollout["old_values"]
    rewards = rollout["rewards"]

    advantages, returns = compute_gae_returns(
        rewards=rewards,
        values=old_values,
        gamma=GAMMA,
        lam=GAE_LAMBDA,
    )
    advantages = normalize_array(advantages)

    n = states.shape[0]
    indices = np.arange(n)
    last_loss = float("nan")

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)

        for start in range(0, n, PPO_BATCH_SIZE):
            batch_idx = indices[start:start + PPO_BATCH_SIZE]

            states_t = tf.convert_to_tensor(states[batch_idx], dtype=tf.float32)
            masks_t = tf.convert_to_tensor(masks[batch_idx], dtype=tf.float32)
            actions_t = tf.convert_to_tensor(actions[batch_idx], dtype=tf.int32)
            old_log_probs_t = tf.convert_to_tensor(old_log_probs[batch_idx], dtype=tf.float32)
            advantages_t = tf.convert_to_tensor(advantages[batch_idx], dtype=tf.float32)
            returns_t = tf.convert_to_tensor(returns[batch_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits, values = policy(states_t, training=True)
                values = tf.squeeze(values, axis=-1)

                masked = masked_logits(logits, masks_t)
                log_probs = tf.nn.log_softmax(masked, axis=-1)
                probs = tf.nn.softmax(masked, axis=-1)

                idx = tf.stack([tf.range(tf.shape(actions_t)[0]), actions_t], axis=1)
                new_log_probs = tf.gather_nd(log_probs, idx)

                ratio = tf.exp(new_log_probs - old_log_probs_t)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)

                policy_loss_1 = ratio * advantages_t
                policy_loss_2 = clipped_ratio * advantages_t
                policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

                value_loss = tf.reduce_mean(tf.square(returns_t - values))

                entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))

                total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_BETA * entropy

            grads = tape.gradient(total_loss, policy.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            optimizer.apply_gradients(zip(grads, policy.trainable_variables))
            last_loss = float(total_loss.numpy())

    return last_loss


# -------------------------
# Training artifacts (CSV + plots)
# -------------------------
def save_training_artifacts(
    *,
    reward_hist: list[float],
    win_hist: list[float],
    step_hist: list[float],
    loss_hist: list[float],
    model_dir: Path,
    level: str,
    timestamp_tag: str,
) -> dict[str, str]:
    n = len(reward_hist)
    if n == 0:
        return {}

    curve_dir = model_dir / CURVE_ARTIFACTS_DIR
    curve_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{level}_{timestamp_tag}"

    csv_path = curve_dir / f"{stem}_episode_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "won", "steps", "loss"])
        writer.writeheader()
        for i in range(n):
            loss_val = loss_hist[i]
            writer.writerow({
                "episode": i + 1,
                "reward": reward_hist[i],
                "won": int(win_hist[i]),
                "steps": step_hist[i],
                "loss": "" if not np.isfinite(loss_val) else loss_val,
            })

    plot_path = curve_dir / f"{stem}_training_curves.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = np.arange(1, n + 1)
        window = min(200, max(10, n // 20))

        def rolling(data: list[float], use_nanmean: bool = False) -> np.ndarray:
            arr = np.array(data, dtype=np.float64)
            out = np.empty(n, dtype=np.float64)
            fn = np.nanmean if use_nanmean else np.mean
            for i in range(n):
                s = max(0, i - window + 1)
                out[i] = fn(arr[s:i + 1])
            return out

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"RL Training — {level}  (rolling window={window})", fontsize=13)

        axes[0].plot(episodes, reward_hist, alpha=0.15, linewidth=0.6)
        axes[0].plot(episodes, rolling(reward_hist), linewidth=1.8)
        axes[0].set_ylabel("Episode reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, win_hist, alpha=0.1, linewidth=0.6)
        axes[1].plot(episodes, rolling(win_hist), linewidth=1.8)
        axes[1].set_ylabel("Win rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

        finite_loss = [x if np.isfinite(x) else np.nan for x in loss_hist]
        axes[2].plot(episodes, finite_loss, alpha=0.15, linewidth=0.6)
        axes[2].plot(episodes, rolling(loss_hist, use_nanmean=True), linewidth=1.8)
        axes[2].set_ylabel("PPO loss")
        axes[2].set_xlabel("Episode")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not save matplotlib plots ({exc}). CSV was still saved.")
        return {"csv": str(csv_path), "plot": ""}

    return {"csv": str(csv_path), "plot": str(plot_path)}


# -------------------------
# Training
# -------------------------
def train(
    level: str,
    model_dir: Path,
    episodes: int,
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

    if architecture_mismatch:
        for stale_path in (model_path, best_metrics_path):
            if stale_path.exists():
                try:
                    stale_path.unlink()
                    print(f"Removed stale file: {stale_path}")
                except OSError as exc:
                    print(f"Could not remove {stale_path} ({exc})")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    previous_best = load_json(best_metrics_path)
    prev_best_win_rate = float(previous_best["best_win_rate"]) if previous_best else -1.0
    best_eval_win_rate = prev_best_win_rate

    reward_hist: list[float] = []
    win_hist: list[float] = []
    step_hist: list[float] = []
    loss_hist: list[float] = []

    print(
        f"level={level}, board={rows}x{cols}, episodes={episodes}, "
    )

    with tqdm(range(1, episodes + 1), desc="Train", unit="ep") as pbar:
        for episode in pbar:
            rollout = play_episode(
                level=level,
                policy=policy,
                predictor=predictor,
                safe_start=safe_start,
                difficulty_seed=int(np.random.randint(0, 2**31 - 1)),
            )

            loss_value = ppo_update(policy, optimizer, rollout)

            reward_hist.append(float(rollout["reward"]))
            win_hist.append(1.0 if rollout["won"] else 0.0)
            step_hist.append(float(rollout["steps"]))
            loss_hist.append(loss_value)

            eval_window = min(WIN_RATE_WINDOW, len(win_hist))
            current_wr = float(np.mean(win_hist[-eval_window:]))
            pbar.set_postfix(
                won="T" if rollout["won"] else "F",
                wr=f"{current_wr:.3f}",
                loss=f"{loss_value:.4f}",
            )

            if episode % SAVE_EVERY_EPISODES == 0 or episode == episodes:
                eval_window = min(WIN_RATE_WINDOW, len(win_hist))
                current_wr = float(np.mean(win_hist[-eval_window:]))

                now = datetime.now().isoformat(timespec="seconds")
                run_metrics = {
                    "timestamp": now,
                    "level": level,
                    "model_dir": str(model_dir),
                    "model_file": model_file,
                    "architecture_mismatch": bool(architecture_mismatch),
                    "base_channels": base_channels,
                    "episodes_requested": episodes,
                    "episodes_completed": episode,
                    "best_win_rate": best_eval_win_rate,
                    "current_win_rate": current_wr,
                    "recent_avg_reward": float(np.mean(reward_hist[-eval_window:])),
                    "recent_avg_steps": float(np.mean(step_hist[-eval_window:])),
                    "recent_avg_loss": float(np.nanmean(loss_hist[-eval_window:])),
                }
                save_json(last_metrics_path, run_metrics)

                if current_wr > best_eval_win_rate + MIN_IMPROVEMENT:
                    best_eval_win_rate = current_wr
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    policy.save(model_path)
                    save_json(best_metrics_path, {**run_metrics, "model_path": str(model_path)})
                    print(f"  >> Policy improved: win_rate={current_wr:.4f} saved -> {model_path}")

    artifact_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = save_training_artifacts(
        reward_hist=reward_hist,
        win_hist=win_hist,
        step_hist=step_hist,
        loss_hist=loss_hist,
        model_dir=model_dir,
        level=level,
        timestamp_tag=artifact_stamp,
    )
    if artifacts.get("csv"):
        print(f"Saved episode CSV: {artifacts['csv']}")
    if artifacts.get("plot"):
        print(f"Saved training curves: {artifacts['plot']}")

    final_metrics = load_json(last_metrics_path)
    if final_metrics is None:
        final_metrics = {
            "level": level,
            "best_win_rate": best_eval_win_rate,
            "episodes_completed": len(reward_hist),
        }
    return final_metrics


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":


    LEVEL_CONFIGS: dict[str, dict[str, Any]] = {
        "easy": {
            "model_dir": Path("models/RL/easy"),
            "model_file": "policy.keras",
            "episodes": 1024,
            "safe_start": True,
        },
        "intermediate": {
            "model_dir": Path("models/RL/intermediate"),
            "model_file": "policy.keras",
            "episodes": 1024,
            "safe_start": True,
        },
        "hard": {
            "model_dir": Path("models/RL/hard"),
            "model_file": "policy.keras",
            "episodes": 1024,
            "safe_start": True,
        },
    }

    cfg = LEVEL_CONFIGS[ACTIVE_LEVEL]
    arch = LEVEL_POLICY_CONFIGS[ACTIVE_LEVEL]

    train(
        level=ACTIVE_LEVEL,
        model_dir=cfg["model_dir"],
        episodes=cfg["episodes"],
        model_file=cfg["model_file"],
        base_channels=arch["base_channels"],
        conv_layers=arch["conv_layers"],
        body_dense_layers=arch["body_dense_layers"],
        head_dense_layers=arch["head_dense_layers"],
        safe_start=cfg["safe_start"],
    )