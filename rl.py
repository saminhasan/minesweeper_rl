from __future__ import annotations

from pathlib import Path

import tensorflow as tf

LEVEL_POLICY_CONFIGS: dict[str, dict] = {
    "easy": {
        "base_channels": 8,
        "conv_layers": 1,
        "body_dense_layers": 2,
        "head_dense_layers": 1,
    },
    "intermediate": {
        "base_channels": 16,
        "conv_layers": 2,
        "body_dense_layers": 4,
        "head_dense_layers": 2,
    },
    "hard": {
        "base_channels": 32,
        "conv_layers": 4,
        "body_dense_layers": 8,
        "head_dense_layers": 4,
    },
}


def build_policy_model(
    height: int,
    width: int,
    channels: int,
    base_channels: int,
    conv_layers: int,
    body_dense_layers: int,
    head_dense_layers: int,
    level_name: str = "",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(height, width, channels), name="policy_input")

    x = inputs

    for i in range(conv_layers):
        x = tf.keras.layers.Conv2D(
            base_channels,
            4 if i == 0 else 3,
            padding="same",
            use_bias=True,
            name=f"body_conv_{i}",
        )(x)
        x = tf.keras.layers.Activation("swish", name=f"body_conv_{i}_act")(x)

    for i in range(body_dense_layers):
        units = base_channels if i == 0 else max(8, base_channels // 2)
        x = tf.keras.layers.Dense(units, activation="swish", name=f"body_dense_{i}")(x)

    p = x
    for i in range(head_dense_layers):
        p = tf.keras.layers.Dense(
            max(8, base_channels // 4),
            activation="swish",
            name=f"policy_dense_{i}",
        )(p)

    policy_map = tf.keras.layers.Dense(1, name="policy_logits_map")(p)
    logits = tf.keras.layers.Reshape((height * width,), name="policy_logits_flat")(policy_map)

    v = x
    for i in range(head_dense_layers):
        v = tf.keras.layers.Dense(
            max(8, base_channels // 2),
            activation="swish",
            name=f"value_dense_{i}",
        )(v)

    v = tf.keras.layers.GlobalAveragePooling2D(name="value_spatial_mean")(v)
    v = tf.keras.layers.Dense(base_channels, activation="swish", name="value_fc")(v)
    value = tf.keras.layers.Dense(1, name="value_output")(v)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[logits, value],
        name=f"minesweeper_policy_{level_name}",
    )


def build_or_load_policy(
    rows: int,
    cols: int,
    in_channels: int,
    model_path: Path,
    base_channels: int,
    conv_layers: int,
    body_dense_layers: int,
    head_dense_layers: int,
    level_name: str = "",
) -> tuple[tf.keras.Model, bool]:
    architecture_mismatch = False

    if model_path.exists():
        try:
            loaded = tf.keras.models.load_model(model_path, compile=False)
            s_in = loaded.input_shape
            s_out = loaded.output_shape

            out_is_multi = (
                isinstance(s_out, (list, tuple))
                and len(s_out) == 2
                and isinstance(s_out[0], (list, tuple))
            )
            logits_ok = out_is_multi and len(s_out[0]) >= 2 and s_out[0][-1] == rows * cols
            io_ok = s_in[1] == rows and s_in[2] == cols and s_in[3] == in_channels and logits_ok

            # stronger architecture check than before
            conv_count = sum(isinstance(l, tf.keras.layers.Conv2D) for l in loaded.layers)
            dense_count = sum(isinstance(l, tf.keras.layers.Dense) for l in loaded.layers)

            expected_conv_count = conv_layers
            expected_dense_count = body_dense_layers + head_dense_layers + head_dense_layers + 3
            # +3 = policy_logits_map + value_fc + value_output

            arch_ok = conv_count == expected_conv_count and dense_count == expected_dense_count

            if io_ok and arch_ok:
                print(f"Loaded previous policy: {model_path}")
                return loaded, False

            architecture_mismatch = True
            print("Previous policy architecture mismatch; starting fresh.")
        except Exception as exc:
            print(f"Failed to load previous policy ({exc}); starting fresh.")
            architecture_mismatch = True

    model = build_policy_model(
        height=rows,
        width=cols,
        channels=in_channels,
        base_channels=base_channels,
        conv_layers=conv_layers,
        body_dense_layers=body_dense_layers,
        head_dense_layers=head_dense_layers,
        level_name=level_name,
    )
    return model, architecture_mismatch