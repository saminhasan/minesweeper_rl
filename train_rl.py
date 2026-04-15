from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from game_engine import Minesweeper, game_mode


@dataclass
class RLConfig:
    difficulty: str = "hard"
    episodes: int = 99999
    gamma: float = 0.99
    learning_rate: float = 3e-4
    dense_units: int = 480
    dense_layers: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    eval_interval: int = 9
    seed: int = 123

    # Prior source used to build RL observations.
    # "cnn": use pretrained CNN probabilities.
    # "bayesian": use game_engine solver probabilities.
    prior_source: str = "bayesian" # "cnn"

    # Warn when CNN vs Bayesian disagreement gets too large.
    disagreement_threshold: float = 0.25
    disagreement_window: int = 50
    print_disagreement_warning: bool = True

    # Reward shaping
    reward_win: float = 1.0
    reward_lose: float = -1.0
    reward_invalid_action: float = -0.05
    reward_reveal_scale: float = 0.01
    reward_step_penalty: float = -0.001

    # Paths
    model_root: Path = Path("models/cnn")
    out_dir: Path = Path("models/rl")


def find_latest_cnn_model(model_root: Path) -> Path:
    candidates = sorted(model_root.glob("*/model.keras"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No CNN model found under {model_root}. Expected e.g. models/cnn/<run>/model.keras"
        )
    return candidates[-1]


def preprocess_board(board_input: np.ndarray) -> np.ndarray:
    # Same scaling as train_cnn.py: covered=-1.0, clues 0..8 -> 0..1.
    return np.where(board_input < 0, -1.0, board_input / 8.0).astype(np.float32)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def build_actor_critic(
    rows: int,
    cols: int,
    input_channels: int = 3,
    dense_units: int = 480,
    dense_layers: int = 4,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(rows, cols, input_channels), name="obs")


    x = inputs
    for _ in range(max(1, dense_layers)):
        x = tf.keras.layers.Conv2D(dense_units, 3, padding="same", activation="relu")(x)


    policy_map = tf.keras.layers.Conv2D(1, 1, padding="same", name="policy_map")(x)
    policy_logits = tf.keras.layers.Reshape((rows * cols,), name="policy_logits")(policy_map)

    value_x = tf.keras.layers.GlobalAveragePooling2D()(x)
    value_x = tf.keras.layers.Dense(128, activation="relu")(value_x)
    value = tf.keras.layers.Dense(1, name="value")(value_x)

    return tf.keras.Model(inputs=inputs, outputs=[policy_logits, value], name="minesweeper_actor_critic")


def masked_logits(logits: tf.Tensor, legal_mask: tf.Tensor) -> tf.Tensor:
    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
    return tf.where(legal_mask > 0.5, logits, neg_inf)


def prior_maps(board: Minesweeper, cnn_model: tf.keras.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    board_input = board.get_input().astype(np.float32)
    board_scaled = preprocess_board(board_input)

    covered_mask = (board.state == board.states.COVERED).astype(np.float32)

    cnn_in = np.expand_dims(board_scaled, axis=(0, -1))  # (1, H, W, 1)
    cnn_prob = cnn_model.predict(cnn_in, verbose=0)[0, ..., 0].astype(np.float32)
    cnn_prob *= covered_mask

    _, bayes_prob = board.solve_minefield()
    bayes_prob = bayes_prob.astype(np.float32) * covered_mask

    return board_scaled, covered_mask, cnn_prob, bayes_prob


def board_obs_with_prior(
    board: Minesweeper,
    cnn_model: tf.keras.Model,
    prior_source: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    board_scaled, covered_mask, cnn_prob, bayes_prob = prior_maps(board, cnn_model)

    if prior_source == "cnn":
        prior = cnn_prob
    elif prior_source == "bayesian":
        prior = bayes_prob
    else:
        raise ValueError(f"Unsupported prior_source: {prior_source}. Use 'cnn' or 'bayesian'.")

    covered_count = float(np.sum(covered_mask))
    if covered_count > 0:
        disagreement = float(np.sum(np.abs(cnn_prob - bayes_prob) * covered_mask) / covered_count)
    else:
        disagreement = 0.0

    # 3 channels: scaled board, legal/covered mask, selected prior.
    obs = np.stack([board_scaled, covered_mask, prior], axis=-1).astype(np.float32)
    legal_flat = covered_mask.reshape(-1).astype(np.float32)
    return obs, legal_flat, disagreement


def action_to_cell(action: int, cols: int) -> tuple[int, int]:
    return action // cols, action % cols


def run_episode(
    policy_model: tf.keras.Model,
    cnn_model: tf.keras.Model,
    cfg: RLConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    board = Minesweeper(cfg.difficulty, seed=int(rng.integers(0, 2**31 - 1)))
    board.random_safe_reveal()

    rows, cols = board.shape

    obs_list: list[np.ndarray] = []
    legal_mask_list: list[np.ndarray] = []
    action_list: list[int] = []
    reward_list: list[float] = []
    disagreement_list: list[float] = []

    steps = 0
    while not (board.game_over or board.game_won):
        obs, legal_flat, disagreement = board_obs_with_prior(
            board=board,
            cnn_model=cnn_model,
            prior_source=cfg.prior_source,
        )
        obs_batch = np.expand_dims(obs, axis=0)
        legal_batch = np.expand_dims(legal_flat, axis=0)

        logits, _ = policy_model(obs_batch, training=False)
        masked = masked_logits(logits, tf.convert_to_tensor(legal_batch, dtype=tf.float32))

        action = int(tf.random.categorical(masked, num_samples=1)[0, 0].numpy())

        reward = 0.0
        if legal_flat[action] <= 0.5:
            reward = cfg.reward_invalid_action
        else:
            i, j = action_to_cell(action, cols)
            before_revealed = int(np.count_nonzero(board.state == board.states.UNCOVERED))
            board.reveal(int(i), int(j))
            after_revealed = int(np.count_nonzero(board.state == board.states.UNCOVERED))
            revealed_delta = max(0, after_revealed - before_revealed)

            reward += cfg.reward_step_penalty + cfg.reward_reveal_scale * float(revealed_delta)
            if board.game_over:
                reward += cfg.reward_lose
            elif board.game_won:
                reward += cfg.reward_win

        obs_list.append(obs)
        legal_mask_list.append(legal_flat)
        action_list.append(action)
        reward_list.append(float(reward))
        disagreement_list.append(disagreement)

        steps += 1

    rewards = np.asarray(reward_list, dtype=np.float32)
    returns = discounted_returns(rewards, cfg.gamma)

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "legal_mask": np.asarray(legal_mask_list, dtype=np.float32),
        "actions": np.asarray(action_list, dtype=np.int32),
        "returns": returns,
        "episode_reward": float(np.sum(rewards)) if rewards.size else 0.0,
        "episode_disagreement": float(np.mean(disagreement_list)) if disagreement_list else 0.0,
        "steps": steps,
        "won": bool(board.game_won),
    }


def train_step(
    policy_model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: dict[str, Any],
    cfg: RLConfig,
) -> dict[str, float]:
    obs = tf.convert_to_tensor(batch["obs"], dtype=tf.float32)
    legal_mask = tf.convert_to_tensor(batch["legal_mask"], dtype=tf.float32)
    actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
    returns = tf.convert_to_tensor(batch["returns"], dtype=tf.float32)

    if tf.shape(obs)[0] == 0:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    with tf.GradientTape() as tape:
        logits, values = policy_model(obs, training=True)
        values = tf.squeeze(values, axis=-1)

        m_logits = masked_logits(logits, legal_mask)
        log_probs = tf.nn.log_softmax(m_logits, axis=-1)
        probs = tf.nn.softmax(m_logits, axis=-1)

        idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        chosen_log_probs = tf.gather_nd(log_probs, idx)

        advantages = returns - values
        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages) + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std

        policy_loss = -tf.reduce_mean(chosen_log_probs * tf.stop_gradient(advantages_norm))
        value_loss = tf.reduce_mean(tf.keras.losses.huber(returns, values))
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))

        total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    grads = tape.gradient(total_loss, policy_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

    return {
        "total_loss": float(total_loss.numpy()),
        "policy_loss": float(policy_loss.numpy()),
        "value_loss": float(value_loss.numpy()),
        "entropy": float(entropy.numpy()),
    }


def main() -> None:
    cfg = RLConfig()

    if cfg.difficulty not in game_mode:
        raise ValueError(f"Unknown difficulty: {cfg.difficulty}")
    if cfg.prior_source not in ("cnn", "bayesian"):
        raise ValueError("prior_source must be 'cnn' or 'bayesian'")


    tf.keras.utils.set_random_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    cnn_path = find_latest_cnn_model(cfg.model_root)
    print(f"Using CNN prior model: {cnn_path}")
    cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
    cnn_model.trainable = False

    rows = game_mode[cfg.difficulty]["rows"]
    cols = game_mode[cfg.difficulty]["columns"]

    cnn_input_shape = cnn_model.input_shape
    if len(cnn_input_shape) != 4 or cnn_input_shape[1] != rows or cnn_input_shape[2] != cols:
        raise ValueError(
            "CNN input shape and RL difficulty do not match: "
            f"cnn expects {cnn_input_shape[1:3]}, "
            f"difficulty '{cfg.difficulty}' is {(rows, cols)}"
        )

    policy_model = build_actor_critic(
        rows=rows,
        cols=cols,
        input_channels=3,
        dense_units=cfg.dense_units,
        dense_layers=cfg.dense_layers,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.out_dir / f"rl_{cfg.difficulty}_{tf.timestamp().numpy():.0f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training difficulty: {cfg.difficulty}")
    print(f"Board: {rows}x{cols}")
    print(f"Episodes: {cfg.episodes}")
    print(f"Network:(dense_layers={cfg.dense_layers}, dense_units={cfg.dense_units})")
    print(f"Prior source: {cfg.prior_source}")
    print(
        f"Disagreement warning: threshold={cfg.disagreement_threshold:.3f}, "
        f"window={cfg.disagreement_window}"
    )

    recent_wins: list[int] = []
    recent_rewards: list[float] = []
    recent_disagreement: deque[float] = deque(maxlen=max(1, cfg.disagreement_window))

    best_win_rate = -1.0
    best_path = run_dir / "best_policy.keras"

    for ep in range(1, cfg.episodes + 1):
        batch = run_episode(policy_model, cnn_model, cfg, rng)
        losses = train_step(policy_model, optimizer, batch, cfg)

        recent_wins.append(1 if batch["won"] else 0)
        recent_rewards.append(float(batch["episode_reward"]))
        recent_disagreement.append(float(batch["episode_disagreement"]))
        if len(recent_wins) > cfg.eval_interval:
            recent_wins.pop(0)
            recent_rewards.pop(0)

        if ep % cfg.eval_interval == 0 or ep == 1:
            win_rate = float(np.mean(recent_wins)) if recent_wins else 0.0
            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            avg_disagreement = float(np.mean(recent_disagreement)) if recent_disagreement else 0.0

            if cfg.print_disagreement_warning and avg_disagreement > cfg.disagreement_threshold:
                print(
                    "[WARN] CNN/Bayesian disagreement is high: "
                    f"avg={avg_disagreement:.3f} > threshold={cfg.disagreement_threshold:.3f}"
                )

            print(
                f"ep={ep:5d} "
                f"win_rate@{len(recent_wins)}={win_rate:.3f} "
                f"avg_reward={avg_reward:.3f} "
                f"avg_disagree={avg_disagreement:.3f} "
                f"steps={batch['steps']} "
                f"loss={losses['total_loss']:.4f}"
            )

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                policy_model.save(best_path)
                print(f"  saved best policy -> {best_path}")

    final_path = run_dir / "final_policy.keras"
    policy_model.save(final_path)

    print("RL training complete.")
    print(f"Best policy: {best_path}")
    print(f"Final policy: {final_path}")


if __name__ == "__main__":
    main()
