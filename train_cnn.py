from datetime import datetime
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Directory containing .npz training shards.
DATA_DIR = Path("data/hard")
# Used only when TRAIN_ALL_SHARDS is False; picks one sorted .npz file by index.
DATASET_NO = 4
# Epochs for single-file mode (TRAIN_ALL_SHARDS=False).
EPOCHS = 16
# Number of samples per optimizer step. Larger is faster but uses more VRAM/RAM.
BATCH_SIZE = 64
# Fraction of each shard held out for validation.
VALIDATION_SPLIT = 0.1
# Adam learning rate.
LEARNING_RATE = 1e-3
# Prefix for output folder names under models/cnn/.
MODEL_NAME = "cnn_small"
# Base RNG seed for split/shuffle reproducibility.
SEED = 123
# Optional cap per shard. 0 means use all samples in the selected shard(s).
MAX_SAMPLES = 0

# If True, train sequentially across selected shards (memory safer for large data).
TRAIN_ALL_SHARDS = True
# Number of epochs to run on each shard before moving to the next shard.
SHARD_EPOCHS = 2
# Optional cap on number of selected shards. 0 means no cap.
MAX_SHARDS = 0
# Skip shards larger than this compressed file size in MB. 0 disables size filtering.
MAX_COMPRESSED_FILE_MB = 300
# Approximate memory budget for one training chunk (in GB of raw sample arrays).
CHUNK_TARGET_GB = 4.0

# Larger value means covered cells are emphasized more in training loss.
COVERED_WEIGHT = 10.0
UNCOVERED_WEIGHT = 0.1

# Prediction is counted correct when absolute error is <= this threshold.
ACCURACY_TOLERANCE = 0.10
# Early stopping patience in single-file mode.
EARLY_STOPPING_PATIENCE = 5
# ReduceLROnPlateau patience in single-file mode.
LR_PATIENCE = 2
# Floor for learning rate after reductions.
MIN_LEARNING_RATE = 1e-6
# Maximum samples loaded into shuffle buffer (RAM/perf tradeoff).
SHUFFLE_BUFFER_SIZE = 8192
# Extra multiplier for underestimation errors in loss.
UNDERESTIMATION_PENALTY = 2.0


_PRINTED_NPZ_CHUNK_WARNING = False


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


def load_dataset(data_dir: Path, dataset_no: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    file_paths = [p for p in data_dir.iterdir() if p.suffix == ".npz"]
    file_paths = sorted(file_paths)

    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if dataset_no < 0 or dataset_no >= len(file_paths):
        raise IndexError(f"dataset_no {dataset_no} out of range 0..{len(file_paths)-1}")

    data_path = file_paths[dataset_no]
    print(f"Loading dataset: {data_path}")
    x, y, mask, _ = load_dataset_by_path(data_path)
    return x, y, mask, data_path


def load_dataset_by_path(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    global _PRINTED_NPZ_CHUNK_WARNING

    print(f"Loading dataset: {data_path}")
    data = np.load(data_path, allow_pickle=False)

    # Keep original dtypes here to avoid large full-array copies up front.
    x = data["x"]
    y = data["y"]
    mask = data["mask"]

    # np.load(..., mmap_mode='r') does not memory-map arrays stored in compressed .npz.
    # Chunk training below still helps downstream memory, but loading the file itself is whole-array.
    if not _PRINTED_NPZ_CHUNK_WARNING and not isinstance(x, np.memmap):
        print(
            "Note: this .npz is loaded into RAM (not memmapped). "
            "Chunking applies during training/evaluation, not initial file load."
        )
        _PRINTED_NPZ_CHUNK_WARNING = True

    return x, y, mask, data_path


def select_dataset_paths(data_dir: Path, dataset_no: int) -> list[Path]:
    file_paths = sorted([p for p in data_dir.iterdir() if p.suffix == ".npz"])
    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if not TRAIN_ALL_SHARDS:
        if dataset_no < 0 or dataset_no >= len(file_paths):
            raise IndexError(f"dataset_no {dataset_no} out of range 0..{len(file_paths)-1}")
        return [file_paths[dataset_no]]

    selected: list[Path] = []
    for p in file_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        if MAX_COMPRESSED_FILE_MB > 0 and size_mb > MAX_COMPRESSED_FILE_MB:
            print(
                f"Skipping shard (compressed size {size_mb:.1f} MB > {MAX_COMPRESSED_FILE_MB} MB): {p.name}"
            )
            continue
        selected.append(p)
        if MAX_SHARDS > 0 and len(selected) >= MAX_SHARDS:
            break

    if not selected:
        raise RuntimeError(
            "No dataset shards selected. Increase MAX_COMPRESSED_FILE_MB or disable shard limits."
        )

    return selected


def preprocess_x(x: np.ndarray) -> np.ndarray:
    # Keep covered as -1.0, scale uncovered clues (0..8) to [0, 1].
    x_scaled = np.where(x < 0, -1.0, x / 8.0).astype(np.float32)
    return np.expand_dims(x_scaled, axis=-1)


def pack_targets(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # y_true[..., 0] = target probability, y_true[..., 1] = covered-cell mask.
    return np.stack([y, mask], axis=-1).astype(np.float32)


def make_masked_weighted_mse(covered_weight: float, uncovered_weight: float):
    def loss(y_true_packed, y_pred):
        y_true, covered_mask = tf.split(y_true_packed, num_or_size_splits=[1, 1], axis=-1)
        weights = covered_mask * covered_weight + (1.0 - covered_mask) * uncovered_weight
        error = y_pred - y_true
        err2 = tf.square(error)

        # Penalize underestimation (y_pred < y_true) more than overestimation.
        under_mask = tf.cast(error < 0.0, tf.float32)
        asym_penalty = 1.0 + under_mask * (UNDERESTIMATION_PENALTY - 1.0)

        weighted_err = err2 * weights * asym_penalty
        return tf.reduce_sum(weighted_err) / tf.reduce_sum(weights)

    return loss


def target_mae(y_true_packed, y_pred):
    y_true, _ = tf.split(y_true_packed, num_or_size_splits=[1, 1], axis=-1)
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def target_accuracy(y_true_packed, y_pred):
    y_true, _ = tf.split(y_true_packed, num_or_size_splits=[1, 1], axis=-1)
    within_tol = tf.less_equal(tf.abs(y_pred - y_true), ACCURACY_TOLERANCE)
    return tf.reduce_mean(tf.cast(within_tol, tf.float32))


def build_model(height: int, width: int, learning_rate: float, covered_weight: float, uncovered_weight: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(height, width, 1), name="board")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="mine_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="minesweeper_cnn_small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_masked_weighted_mse(covered_weight=covered_weight, uncovered_weight=uncovered_weight),
        metrics=[target_mae, target_accuracy],
    )
    return model


def train_val_split(
    n: int,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_split < 1.0):
        raise ValueError("validation_split must be between 0 and 1")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return train_idx, val_idx


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    indices = np.asarray(indices, dtype=np.int64)
    ds = tf.data.Dataset.from_tensor_slices(indices)

    if training:
        buffer_size = min(SHUFFLE_BUFFER_SIZE, int(indices.shape[0]))
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)

    def _load_sample_np(i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = int(i)

        xi = x[idx].astype(np.float32, copy=False)
        yi = y[idx].astype(np.float32, copy=False)
        mi = mask[idx].astype(np.float32, copy=False)

        # Keep covered as -1.0, scale uncovered clues (0..8) to [0, 1].
        x_scaled = np.where(xi < 0, -1.0, xi / 8.0).astype(np.float32, copy=False)
        x_out = np.expand_dims(x_scaled, axis=-1)

        # y_true[..., 0] = target probability, y_true[..., 1] = covered-cell mask.
        y_out = np.stack([yi, mi], axis=-1).astype(np.float32, copy=False)
        return x_out, y_out

    def _load_sample_tf(i: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x_item, y_item = tf.numpy_function(
            _load_sample_np,
            [i],
            [tf.float32, tf.float32],
        )
        x_item.set_shape((x.shape[1], x.shape[2], 1))
        y_item.set_shape((y.shape[1], y.shape[2], 2))
        return x_item, y_item

    ds = ds.map(_load_sample_tf, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def estimate_sample_bytes(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> int:
    return int(x[0].nbytes + y[0].nbytes + mask[0].nbytes)


def split_indices_into_chunks(indices: np.ndarray, bytes_per_sample: int, target_gb: float) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if target_gb <= 0:
        return [indices]

    target_bytes = int(target_gb * (1024**3))
    samples_per_chunk = max(1, target_bytes // max(1, bytes_per_sample))

    chunks: list[np.ndarray] = []
    for start in range(0, int(indices.shape[0]), samples_per_chunk):
        chunks.append(indices[start : start + samples_per_chunk])
    return chunks


def masked_mae_over_indices(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    covered_err = 0.0
    covered_count = 0.0
    uncovered_err = 0.0
    uncovered_count = 0.0

    for start in range(0, int(indices.shape[0]), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = preprocess_x(x[batch_idx])
        y_batch = y[batch_idx].astype(np.float32, copy=False)
        m_batch = mask[batch_idx].astype(np.float32, copy=False)

        pred_batch = model.predict(x_batch, batch_size=batch_size, verbose=0)[..., 0]

        abs_err = np.abs(y_batch - pred_batch)
        covered_err += float(np.sum(abs_err * m_batch))
        covered_count += float(np.sum(m_batch))

        um = 1.0 - m_batch
        uncovered_err += float(np.sum(abs_err * um))
        uncovered_count += float(np.sum(um))

    covered_mae = float("nan") if covered_count <= 0 else (covered_err / covered_count)
    uncovered_mae = float("nan") if uncovered_count <= 0 else (uncovered_err / uncovered_count)
    return covered_mae, uncovered_mae


def masked_mae_numpy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    # Inputs are (B, H, W). mask is 1 for covered cells.
    denom = np.sum(mask)
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred) * mask) / denom)


def save_training_curves(history: tf.keras.callbacks.History, run_dir: Path) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_over_epochs.png", dpi=150)
    plt.close()

    acc_key = "target_accuracy"
    val_acc_key = "val_target_accuracy"
    if acc_key in history.history:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history.history[acc_key], label="Train Accuracy")
        if val_acc_key in history.history:
            plt.plot(epochs, history.history[val_acc_key], label="Val Accuracy")
        plt.title(f"Accuracy Over Epochs (|error| <= {ACCURACY_TOLERANCE})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "accuracy_over_epochs.png", dpi=150)
        plt.close()


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    print_gpu_status()

    dataset_paths = select_dataset_paths(DATA_DIR, DATASET_NO)

    if TRAIN_ALL_SHARDS:
        print("Shard mode enabled.")
        print(f"Selected {len(dataset_paths)} shard(s):")
        for i, p in enumerate(dataset_paths):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [{i}] {p} ({size_mb:.1f} MB compressed)")

        x0, y0, _m0, _ = load_dataset_by_path(dataset_paths[0])
        model = build_model(
            height=x0.shape[1],
            width=x0.shape[2],
            learning_rate=LEARNING_RATE,
            covered_weight=COVERED_WEIGHT,
            uncovered_weight=UNCOVERED_WEIGHT,
        )
        del x0, y0, _m0
        gc.collect()

        model.summary()

        run_dir = Path("models/cnn") / f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        agg_history: dict[str, list[float]] = {
            "loss": [],
            "val_loss": [],
            "target_accuracy": [],
            "val_target_accuracy": [],
        }
        shard_metrics: list[dict[str, float | int | str]] = []

        for shard_i, shard_path in enumerate(dataset_paths):
            x, y, mask, data_path = load_dataset_by_path(shard_path)

            if MAX_SAMPLES > 0:
                limit = min(MAX_SAMPLES, x.shape[0])
                x = x[:limit]
                y = y[:limit]
                mask = mask[:limit]
                print(f"Using first {limit} samples due to MAX_SAMPLES")

            train_idx, val_idx = train_val_split(
                n=x.shape[0],
                val_split=VALIDATION_SPLIT,
                seed=SEED + shard_i,
            )

            bytes_per_sample = estimate_sample_bytes(x, y, mask)
            train_chunks = split_indices_into_chunks(
                indices=train_idx,
                bytes_per_sample=bytes_per_sample,
                target_gb=CHUNK_TARGET_GB,
            )

            val_ds = make_tf_dataset(
                x=x,
                y=y,
                mask=mask,
                indices=val_idx,
                batch_size=BATCH_SIZE,
                training=False,
                seed=SEED + shard_i,
            )

            print(f"Shard {shard_i + 1}/{len(dataset_paths)}")
            print(f"x shape: {x.shape}, dtype={x.dtype}")
            print(f"y target shape: {y.shape}, dtype={y.dtype}")
            print(f"mask shape: {mask.shape}, dtype={mask.dtype}")
            print(f"train samples={train_idx.shape[0]}, val samples={val_idx.shape[0]}")
            print(
                f"chunking: target={CHUNK_TARGET_GB:.2f} GB, sample_bytes={bytes_per_sample}, chunks={len(train_chunks)}"
            )
            print(
                f"tf.data pipeline: shuffle={SHUFFLE_BUFFER_SIZE}, batch={BATCH_SIZE}, prefetch=AUTOTUNE"
            )

            callbacks: list[tf.keras.callbacks.Callback] = [
                tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
            ]

            last_history: tf.keras.callbacks.History | None = None
            for chunk_i, chunk_idx in enumerate(train_chunks):
                train_ds = make_tf_dataset(
                    x=x,
                    y=y,
                    mask=mask,
                    indices=chunk_idx,
                    batch_size=BATCH_SIZE,
                    training=True,
                    seed=SEED + shard_i + chunk_i,
                )

                print(
                    f"  Chunk {chunk_i + 1}/{len(train_chunks)}: {chunk_idx.shape[0]} train samples"
                )
                history = model.fit(
                    train_ds,
                    epochs=SHARD_EPOCHS,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    verbose=1,
                )
                last_history = history

                for k in agg_history:
                    if k in history.history:
                        agg_history[k].extend([float(v) for v in history.history[k]])

                del train_ds
                gc.collect()

            if last_history is None:
                raise RuntimeError("No training chunks were produced for this shard")

            covered_mae, uncovered_mae = masked_mae_over_indices(
                model=model,
                x=x,
                y=y,
                mask=mask,
                indices=val_idx,
                batch_size=BATCH_SIZE,
            )

            shard_metrics.append(
                {
                    "shard_index": int(shard_i),
                    "dataset_path": str(data_path),
                    "train_samples": int(train_idx.shape[0]),
                    "val_samples": int(val_idx.shape[0]),
                    "chunk_count": int(len(train_chunks)),
                    "final_train_loss": float(last_history.history["loss"][-1]),
                    "final_val_loss": float(last_history.history["val_loss"][-1]),
                    "covered_mae": covered_mae,
                    "uncovered_mae": uncovered_mae,
                    "final_train_accuracy": float(last_history.history.get("target_accuracy", [float("nan")])[-1]),
                    "final_val_accuracy": float(last_history.history.get("val_target_accuracy", [float("nan")])[-1]),
                }
            )

            del val_ds, x, y, mask
            gc.collect()

        model_path = run_dir / "model.keras"
        model.save(model_path)

        save_training_curves(SimpleNamespace(history=agg_history), run_dir)

        metrics_payload = {
            "train_all_shards": True,
            "epochs_per_shard": int(SHARD_EPOCHS),
            "shard_count": int(len(dataset_paths)),
            "final_train_loss": float(agg_history["loss"][-1]) if agg_history["loss"] else float("nan"),
            "final_val_loss": float(agg_history["val_loss"][-1]) if agg_history["val_loss"] else float("nan"),
            "final_train_accuracy": float(agg_history["target_accuracy"][-1])
            if agg_history["target_accuracy"]
            else float("nan"),
            "final_val_accuracy": float(agg_history["val_target_accuracy"][-1])
            if agg_history["val_target_accuracy"]
            else float("nan"),
            "accuracy_tolerance": float(ACCURACY_TOLERANCE),
            "underestimation_penalty": float(UNDERESTIMATION_PENALTY),
            "covered_weight": float(COVERED_WEIGHT),
            "uncovered_weight": float(UNCOVERED_WEIGHT),
            "shards": shard_metrics,
        }

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        print("Training complete.")
        print(f"Model saved to: {model_path}")
        print(f"Shards trained: {len(dataset_paths)}")
        if agg_history["loss"]:
            print(f"Final train loss: {agg_history['loss'][-1]:.6f}")
        if agg_history["val_loss"]:
            print(f"Final val loss: {agg_history['val_loss'][-1]:.6f}")
        if agg_history["target_accuracy"]:
            print(f"Final train accuracy: {agg_history['target_accuracy'][-1]:.6f}")
        if agg_history["val_target_accuracy"]:
            print(f"Final val accuracy: {agg_history['val_target_accuracy'][-1]:.6f}")
        print(f"Saved curves: {run_dir / 'loss_over_epochs.png'}")
        print(f"Saved curves: {run_dir / 'accuracy_over_epochs.png'}")
        print(f"Artifacts saved in: {run_dir}")
        return

    x, y, mask, data_path = load_dataset(DATA_DIR, DATASET_NO)

    if MAX_SAMPLES > 0:
        limit = min(MAX_SAMPLES, x.shape[0])
        x = x[:limit]
        y = y[:limit]
        mask = mask[:limit]
        print(f"Using first {limit} samples due to MAX_SAMPLES")

    train_idx, val_idx = train_val_split(
        n=x.shape[0],
        val_split=VALIDATION_SPLIT,
        seed=SEED,
    )

    bytes_per_sample = estimate_sample_bytes(x, y, mask)
    train_chunks = split_indices_into_chunks(
        indices=train_idx,
        bytes_per_sample=bytes_per_sample,
        target_gb=CHUNK_TARGET_GB,
    )
    val_ds = make_tf_dataset(
        x=x,
        y=y,
        mask=mask,
        indices=val_idx,
        batch_size=BATCH_SIZE,
        training=False,
        seed=SEED,
    )

    print(f"x shape: {x.shape}, dtype={x.dtype}")
    print(f"y target shape: {y.shape}, dtype={y.dtype}")
    print(f"mask shape: {mask.shape}, dtype={mask.dtype}")
    print(f"train samples={train_idx.shape[0]}, val samples={val_idx.shape[0]}")
    print(
        f"chunking: target={CHUNK_TARGET_GB:.2f} GB, sample_bytes={bytes_per_sample}, chunks={len(train_chunks)}"
    )
    print(f"tf.data pipeline: shuffle={SHUFFLE_BUFFER_SIZE}, batch={BATCH_SIZE}, prefetch=AUTOTUNE")

    model = build_model(
        height=x.shape[1],
        width=x.shape[2],
        learning_rate=LEARNING_RATE,
        covered_weight=COVERED_WEIGHT,
        uncovered_weight=UNCOVERED_WEIGHT,
    )
    model.summary()

    run_dir = Path("models/cnn") / f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = [
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
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history: tf.keras.callbacks.History | None = None
    for epoch_i in range(EPOCHS):
        print(f"Epoch group {epoch_i + 1}/{EPOCHS}")
        for chunk_i, chunk_idx in enumerate(train_chunks):
            train_ds = make_tf_dataset(
                x=x,
                y=y,
                mask=mask,
                indices=chunk_idx,
                batch_size=BATCH_SIZE,
                training=True,
                seed=SEED + epoch_i + chunk_i,
            )
            print(f"  Chunk {chunk_i + 1}/{len(train_chunks)}: {chunk_idx.shape[0]} train samples")
            history = model.fit(
                train_ds,
                epochs=1,
                validation_data=val_ds,
                callbacks=callbacks,
                verbose=1,
            )
            del train_ds
            gc.collect()

    if history is None:
        raise RuntimeError("No training chunks were produced")

    model_path = run_dir / "model.keras"
    model.save(model_path)

    covered_mae, uncovered_mae = masked_mae_over_indices(
        model=model,
        x=x,
        y=y,
        mask=mask,
        indices=val_idx,
        batch_size=BATCH_SIZE,
    )

    save_training_curves(history, run_dir)

    metrics_payload = {
        "dataset_path": str(data_path),
        "train_samples": int(train_idx.shape[0]),
        "val_samples": int(val_idx.shape[0]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "covered_mae": covered_mae,
        "uncovered_mae": uncovered_mae,
        "final_train_accuracy": float(history.history.get("target_accuracy", [float("nan")])[-1]),
        "final_val_accuracy": float(history.history.get("val_target_accuracy", [float("nan")])[-1]),
        "accuracy_tolerance": float(ACCURACY_TOLERANCE),
        "underestimation_penalty": float(UNDERESTIMATION_PENALTY),
        "covered_weight": float(COVERED_WEIGHT),
        "uncovered_weight": float(UNCOVERED_WEIGHT),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_path}")
    print(f"Final train loss: {history.history['loss'][-1]:.6f}")
    if "val_loss" in history.history:
        print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
    if "target_accuracy" in history.history:
        print(f"Final train accuracy: {history.history['target_accuracy'][-1]:.6f}")
    if "val_target_accuracy" in history.history:
        print(f"Final val accuracy: {history.history['val_target_accuracy'][-1]:.6f}")
    print(f"Validation covered-cell MAE: {covered_mae:.6f}")
    print(f"Validation uncovered-cell MAE: {uncovered_mae:.6f}")
    print(f"Underestimation penalty factor: {UNDERESTIMATION_PENALTY}")
    print(f"Saved curves: {run_dir / 'loss_over_epochs.png'}")
    print(f"Saved curves: {run_dir / 'accuracy_over_epochs.png'}")
    print(f"Artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()