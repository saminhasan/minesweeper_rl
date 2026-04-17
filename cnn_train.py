from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


# -------------------------
# Hyperparameters
# -------------------------
VALIDATION_SPLIT        = 0.1
LEARNING_RATE           = 1e-3
SEED                    = 123

SHUFFLE_BUFFER          = 8192
COVERED_WEIGHT          = 2.0
UNCOVERED_WEIGHT        = 1.0
UNDERESTIMATION_PENALTY = 2.0
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE             = 2
MIN_LEARNING_RATE       = 1e-6
MIN_IMPROVEMENT         = 1e-6

BASE_CHANNELS           = 64
DROPOUT_RATE            = 0.10

MODEL_FILE              = "model.keras"
BEST_METRICS_FILE       = "best_metrics.json"
LAST_RUN_METRICS_FILE   = "last_run_metrics.json"
CURVE_ARTIFACTS_DIR     = "training_curves"


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


def select_run_dir(level_data_dir: Path, run_name: str) -> Path:
    if not level_data_dir.exists():
        raise FileNotFoundError(f"Level dataset directory not found: {level_data_dir}")
    if run_name:
        run_dir = level_data_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted([p for p in level_data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {level_data_dir}")
    return candidates[-1]


def build_input_channels(x: np.ndarray, covered_mask: np.ndarray) -> np.ndarray:
    """Build 4-channel feature tensor (numpy version, used by train_rl.py)."""
    x_f = x.astype(np.float32, copy=False)
    covered = covered_mask.astype(np.float32, copy=False)

    board_scaled = np.where(x_f < 0.0, -1.0, x_f / 8.0).astype(np.float32, copy=False)
    revealed_clue = np.where(covered > 0.5, 0.0, np.clip(x_f, 0.0, 8.0) / 8.0).astype(np.float32, copy=False)
    uncovered = (covered <= 0.5).astype(np.float32, copy=False)

    padded = np.pad(uncovered, ((0, 0), (1, 1), (1, 1)), mode="constant")
    near_uncovered = (
        padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:] +
        padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:] +
        padded[:, 2:, :-2] + padded[:, 2:, 1:-1] + padded[:, 2:, 2:]
    )
    frontier_mask = ((covered > 0.5) & (near_uncovered > 0.0)).astype(np.float32, copy=False)

    return np.stack([board_scaled, covered, revealed_clue, frontier_mask], axis=-1)


def pack_targets(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    target[..., 0] = P(mine)
    target[..., 1] = P(safe)
    target[..., 2] = covered mask
    """
    y_f = y.astype(np.float32, copy=False)
    m_f = mask.astype(np.float32, copy=False)
    return np.stack([y_f, 1.0 - y_f, m_f], axis=-1)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_loss_curve_artifacts(
    *,
    history: tf.keras.callbacks.History,
    level: str,
    model_dir: Path,
    run_dir: Path,
    timestamp_tag: str,
) -> dict[str, str]:
    train_loss = np.asarray(history.history.get("loss", []), dtype=np.float32)
    val_loss = np.asarray(history.history.get("val_loss", []), dtype=np.float32)
    if train_loss.size == 0 or val_loss.size == 0:
        return {}

    epoch_count = min(train_loss.size, val_loss.size)
    train_loss = train_loss[:epoch_count]
    val_loss = val_loss[:epoch_count]
    epochs = np.arange(1, epoch_count + 1, dtype=np.int32)

    curve_dir = model_dir / CURVE_ARTIFACTS_DIR
    curve_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{level}_{run_dir.name}_{timestamp_tag}"

    npz_path = curve_dir / f"{stem}_loss_data.npz"
    train_plot_path = curve_dir / f"{stem}_train_loss.png"
    val_plot_path = curve_dir / f"{stem}_val_loss.png"

    np.savez_compressed(
        npz_path,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        level=np.array(level),
        run_dir=np.array(str(run_dir)),
        model_dir=np.array(str(model_dir)),
    )

    try:
        import matplotlib
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, color="tab:blue", marker="o", markersize=3, linewidth=1.5)
        ax.set_title(f"{level.capitalize()} Train Loss vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(train_plot_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, val_loss, color="tab:orange", marker="o", markersize=3, linewidth=1.5)
        ax.set_title(f"{level.capitalize()} Val Loss vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(val_plot_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Could not save matplotlib loss plots ({exc}). NPZ data was still saved: {npz_path}")
        return {
            "npz": str(npz_path),
            "train_plot": "",
            "val_plot": "",
        }

    return {
        "npz": str(npz_path),
        "train_plot": str(train_plot_path),
        "val_plot": str(val_plot_path),
    }


def make_tf_dataset(
    bin_path: Path,
    rows: int,
    cols: int,
    n_planes: int,
    num_samples: int,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, int, int, int]:
    """Stream train/val pipelines from monolithic binary file."""
    record_bytes = n_planes * rows * cols * 8  # float64 per snapshot

    def parse_and_build(raw: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        planes = tf.cast(
            tf.reshape(tf.io.decode_raw(raw, tf.float64), (n_planes, rows, cols)),
            tf.float32,
        )
        x = planes[0]       # covered=-1, uncovered=0..8
        y = planes[1]       # Bayesian mine probability
        covered = planes[2] # 1.0=covered, 0.0=uncovered

        board_scaled = tf.where(x < 0.0, tf.fill((rows, cols), -1.0), x / 8.0)
        revealed_clue = tf.where(covered > 0.5, tf.zeros((rows, cols)), tf.clip_by_value(x, 0.0, 8.0) / 8.0)
        uncovered_f = tf.cast(covered <= 0.5, tf.float32)

        kernel = tf.ones((3, 3, 1, 1), tf.float32)
        u4d = uncovered_f[tf.newaxis, :, :, tf.newaxis]
        near = tf.nn.conv2d(u4d, kernel, strides=1, padding="SAME")[0, :, :, 0] - uncovered_f
        frontier = tf.cast(tf.math.logical_and(covered > 0.5, near > 0.0), tf.float32)

        features = tf.stack([board_scaled, covered, revealed_clue, frontier], axis=-1)

        safe = 1.0 - y
        target = tf.stack([y, safe, covered], axis=-1)
        return features, target

    n_val = max(1, int(num_samples * VALIDATION_SPLIT))
    n_train = num_samples - n_val
    buf = min(SHUFFLE_BUFFER, n_train)
    train_steps = max(1, int(np.ceil(n_train / batch_size)))
    val_steps = max(1, int(np.ceil(n_val / batch_size)))

    ds_raw = tf.data.FixedLengthRecordDataset(str(bin_path), record_bytes)

    val_ds = (
        ds_raw.take(n_val)
        .map(parse_and_build, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds = (
        ds_raw.skip(n_val)
        .shuffle(buf, seed=SEED, reshuffle_each_iteration=True)
        .map(parse_and_build, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, n_train, n_val, train_steps, val_steps


def make_masked_weighted_dual_mse(covered_weight: float, uncovered_weight: float):
    """
    y_true_packed[..., 0] = true P(mine)
    y_true_packed[..., 1] = true P(safe)
    y_true_packed[..., 2] = covered mask

    y_pred[..., 0] = pred P(mine)
    y_pred[..., 1] = pred P(safe)
    """
    def loss(y_true_packed, y_pred):
        y_true = y_true_packed[..., :2]
        covered_mask = y_true_packed[..., 2:3]

        weights = covered_mask * covered_weight + (1.0 - covered_mask) * uncovered_weight
        error = y_pred - y_true
        sq_err = tf.square(error)

        mine_error = error[..., 0:1]
        under_mask = tf.cast(mine_error < 0.0, tf.float32)
        asym_penalty = 1.0 + under_mask * (UNDERESTIMATION_PENALTY - 1.0)

        ch_weights = tf.concat([asym_penalty, tf.ones_like(asym_penalty)], axis=-1)
        weighted = sq_err * weights * ch_weights

        denom = tf.reduce_sum(weights) * 2.0
        return tf.reduce_sum(weighted) / (denom + tf.keras.backend.epsilon())

    return loss


def masked_mine_mae_metric(y_true_packed, y_pred):
    y_true_mine = y_true_packed[..., 0:1]
    covered_mask = y_true_packed[..., 2:3]
    pred_mine = y_pred[..., 0:1]

    abs_err = tf.abs(pred_mine - y_true_mine)
    denom = tf.reduce_sum(covered_mask)
    return tf.reduce_sum(abs_err * covered_mask) / (denom + tf.keras.backend.epsilon())


def masked_safe_mae_metric(y_true_packed, y_pred):
    y_true_safe = y_true_packed[..., 1:2]
    covered_mask = y_true_packed[..., 2:3]
    pred_safe = y_pred[..., 1:2]

    abs_err = tf.abs(pred_safe - y_true_safe)
    denom = tf.reduce_sum(covered_mask)
    return tf.reduce_sum(abs_err * covered_mask) / (denom + tf.keras.backend.epsilon())


def residual_dilated_block(x: tf.Tensor, channels: int, dilation_rate: int) -> tf.Tensor:
    shortcut = x
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", dilation_rate=dilation_rate, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", dilation_rate=dilation_rate, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("swish")(x)
    return x


def build_model(
    height: int,
    width: int,
    base_channels: int = BASE_CHANNELS,
    dropout_rate: float = DROPOUT_RATE,
    dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
    level_name: str = "",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(height, width, 4), name="board_features")

    x = tf.keras.layers.Conv2D(base_channels, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)

    for rate in dilation_rates:
        x = residual_dilated_block(x, base_channels, dilation_rate=rate)

    x = tf.keras.layers.Conv2D(base_channels * 2, 1, padding="same", activation="swish")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(base_channels, 1, padding="same", activation="swish")(x)

    logits = tf.keras.layers.Conv2D(2, 1, padding="same", activation=None, name="mine_safe_logits")(x)
    outputs = tf.keras.layers.Softmax(axis=-1, name="mine_safe_prob")(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"minesweeper_cnn_{level_name}")


def compile_model(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=make_masked_weighted_dual_mse(COVERED_WEIGHT, UNCOVERED_WEIGHT),
        metrics=[masked_mine_mae_metric, masked_safe_mae_metric],
    )


def build_or_load_model(
    rows: int,
    cols: int,
    model_path: Path,
    base_channels: int = BASE_CHANNELS,
    dropout_rate: float = DROPOUT_RATE,
    dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
    level_name: str = "",
) -> tuple[tf.keras.Model, bool]:
    architecture_mismatch = False

    if model_path.exists():
        try:
            loaded = tf.keras.models.load_model(model_path, compile=False)

            s_in = loaded.input_shape
            s_out = loaded.output_shape
            conv_layers = [l for l in loaded.layers if isinstance(l, tf.keras.layers.Conv2D)]
            drop_layers = [l for l in loaded.layers if isinstance(l, tf.keras.layers.Dropout)]

            conv_count_expected = 2 * len(dilation_rates) + 4
            first_conv_ok = bool(conv_layers) and int(conv_layers[0].filters) == int(base_channels)
            conv_count_ok = len(conv_layers) == conv_count_expected
            dropout_ok = len(drop_layers) == 1 and abs(float(drop_layers[0].rate) - float(dropout_rate)) < 1e-9
            io_ok = (
                s_in[1] == rows and s_in[2] == cols and s_in[3] == 4 and
                s_out[1] == rows and s_out[2] == cols and s_out[3] == 2
            )

            if io_ok and first_conv_ok and conv_count_ok and dropout_ok:
                compile_model(loaded)
                print(f"Loaded previous model: {model_path}")
                return loaded, False

            architecture_mismatch = True
            print("Previous model architecture mismatch; starting fresh.")
        except Exception as exc:
            print(f"Failed to load previous model ({exc}); starting fresh.")

    model = build_model(
        rows,
        cols,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
        dilation_rates=dilation_rates,
        level_name=level_name,
    )
    compile_model(model)
    return model, architecture_mismatch


def train(
    level: str,
    data_dir: Path,
    run_name: str,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    model_file: str = MODEL_FILE,
    base_channels: int = BASE_CHANNELS,
    dropout_rate: float = DROPOUT_RATE,
    dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
    level_name: str = "",
) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(SEED)
    print_gpu_status()

    run_dir = select_run_dir(data_dir, run_name)
    print(f"Selected run: {run_dir}")

    meta = load_json(run_dir / "metadata.json")
    if meta is None:
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_planes = int(meta["payload"]["plane_count"])
    num_samples = int(meta["num_samples"])
    bin_path = run_dir / "data.bin"

    print(f"level={level}, samples={num_samples:,}, board={rows}x{cols}, epochs={epochs}, batch={batch_size}")

    train_ds, val_ds, n_train, n_val, train_steps, val_steps = make_tf_dataset(
        bin_path=bin_path,
        rows=rows,
        cols=cols,
        n_planes=n_planes,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    model_path = model_dir / model_file
    best_metrics_path = model_dir / BEST_METRICS_FILE
    last_metrics_path = model_dir / LAST_RUN_METRICS_FILE

    model, architecture_mismatch = build_or_load_model(
        rows,
        cols,
        model_path,
        base_channels=base_channels,
        dropout_rate=dropout_rate,
        dilation_rates=dilation_rates,
        level_name=level_name,
    )

    if architecture_mismatch:
        if model_path.exists():
            try:
                model_path.unlink()
                print(f"Removed mismatched checkpoint: {model_path}")
            except OSError as exc:
                print(f"Could not remove mismatched checkpoint ({exc})")
        if best_metrics_path.exists():
            try:
                best_metrics_path.unlink()
                print(f"Removed stale best metrics: {best_metrics_path}")
            except OSError as exc:
                print(f"Could not remove stale best metrics ({exc})")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=epochs,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
            ),
        ],
    )

    best_val_loss = float(np.min(history.history.get("val_loss", [np.inf])))
    best_train_loss = float(np.min(history.history.get("loss", [np.inf])))
    final_val_mine_mae = float(history.history.get("val_masked_mine_mae_metric", [np.nan])[-1])
    final_val_safe_mae = float(history.history.get("val_masked_safe_mae_metric", [np.nan])[-1])

    previous_best = load_json(best_metrics_path)
    prev_best_val = float(previous_best["best_val_loss"]) if previous_best else float("inf")
    improved = best_val_loss < (prev_best_val - MIN_IMPROVEMENT)

    now = datetime.now().isoformat(timespec="seconds")
    artifact_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_artifacts = save_loss_curve_artifacts(
        history=history,
        level=level,
        model_dir=model_dir,
        run_dir=run_dir,
        timestamp_tag=artifact_stamp,
    )

    run_metrics: dict[str, Any] = {
        "timestamp": now,
        "level": level,
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "model_file": model_file,
        "architecture_mismatch": bool(architecture_mismatch),
        "base_channels": base_channels,
        "dropout_rate": dropout_rate,
        "dilation_rates": list(dilation_rates),
        "num_samples": num_samples,
        "train_samples": n_train,
        "val_samples": n_val,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "final_val_mine_mae": final_val_mine_mae,
        "final_val_safe_mae": final_val_safe_mae,
        "previous_best_val_loss": prev_best_val,
        "improved": bool(improved),
        "loss_curve_npz": curve_artifacts.get("npz", ""),
        "train_loss_plot": curve_artifacts.get("train_plot", ""),
        "val_loss_plot": curve_artifacts.get("val_plot", ""),
    }

    save_json(last_metrics_path, run_metrics)

    if improved:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        save_json(best_metrics_path, {**run_metrics, "model_path": str(model_path)})
        print(f"Model improved: saved -> {model_path}")
        print(f"Best val_loss: {best_val_loss:.6f} (prev {prev_best_val:.6f})")
    else:
        print("Model did not improve: existing best kept.")
        print(f"Current val_loss: {best_val_loss:.6f}, best val_loss: {prev_best_val:.6f}")

    if curve_artifacts:
        print(f"Saved loss data NPZ: {curve_artifacts.get('npz', '')}")
        if curve_artifacts.get("train_plot"):
            print(f"Saved train-loss plot: {curve_artifacts['train_plot']}")
        if curve_artifacts.get("val_plot"):
            print(f"Saved val-loss plot: {curve_artifacts['val_plot']}")

    print(f"Saved last run metrics: {last_metrics_path}")
    return run_metrics


if __name__ == "__main__":
    levels = ["easy", "intermediate", "hard"]
    ACTIVE_LEVEL = levels[2]

    LEVEL_CONFIGS: dict[str, dict[str, Any]] = {
        "easy": {
            "run_name": "",
            "data_dir": Path("data/easy"),
            "model_dir": Path("models/cnn/easy"),
            "model_file": "model.keras",
            "epochs": 32,
            "batch_size": 256,
            "base_channels": 48,
            "dropout_rate": 0.08,
            "dilation_rates": (1, 2, 4),
        },
        "intermediate": {
            "run_name": "",
            "data_dir": Path("data/intermediate"),
            "model_dir": Path("models/cnn/intermediate"),
            "model_file": "model.keras",
            "epochs": 48,
            "batch_size": 128,
            "base_channels": 64,
            "dropout_rate": 0.10,
            "dilation_rates": (1, 2, 4, 8),
        },
        "hard": {
            "run_name": "",
            "data_dir": Path("data/hard"),
            "model_dir": Path("models/cnn/hard"),
            "model_file": "model.keras",
            "epochs": 64,
            "batch_size": 64,
            "base_channels": 96,
            "dropout_rate": 0.12,
            "dilation_rates": (1, 2, 4, 8, 16),
        },
    }

    cfg = LEVEL_CONFIGS[ACTIVE_LEVEL]

    train(
        level=ACTIVE_LEVEL,
        data_dir=cfg["data_dir"],
        run_name=cfg["run_name"],
        model_dir=cfg["model_dir"],
        model_file=cfg["model_file"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        base_channels=cfg["base_channels"],
        dropout_rate=cfg["dropout_rate"],
        dilation_rates=cfg["dilation_rates"],
    )