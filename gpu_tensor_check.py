#!/usr/bin/env python3
"""Simple TensorFlow GPU health and workload check."""

from __future__ import annotations

import argparse
import sys
import time

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random TensorFlow ops on GPU.")
    parser.add_argument("--size", type=int, default=1024, help="Square matrix size")
    parser.add_argument("--steps", type=int, default=5, help="Number of matmul iterations")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tf.random.set_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")

    print(f"TensorFlow: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPUs: {gpus}")

    if not gpus:
        print("FAIL: No GPU detected by TensorFlow", file=sys.stderr)
        return 1

    size = args.size
    steps = args.steps

    # Run a small but real GPU workload: chained matrix multiplications.
    with tf.device("/GPU:0"):
        a = tf.random.uniform((size, size), dtype=tf.float32)
        b = tf.random.uniform((size, size), dtype=tf.float32)

        # Warm-up so first-time kernel compilation/loading does not skew timing.
        c = tf.matmul(a, b)
        _ = c.numpy()

        start = time.perf_counter()
        for _ in range(steps):
            c = tf.matmul(c, b)
        checksum = float(tf.reduce_mean(c).numpy())
        elapsed = time.perf_counter() - start

    print(f"Workload: {steps} matmuls of [{size}x{size}] on /GPU:0")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Per step: {elapsed / steps:.4f}s")
    print(f"Checksum(mean): {checksum:.6f}")
    print("PASS: GPU tensor workload completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
