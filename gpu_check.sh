#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root to verify WSL + TensorFlow GPU readiness.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "rl/bin/activate" ]]; then
  echo "Error: rl/bin/activate not found. Run this from the repo root." >&2
  exit 1
fi

source rl/bin/activate

# Silence TensorFlow startup logs in this health-check script.
export TF_CPP_MIN_LOG_LEVEL="3"
export TF_ENABLE_ONEDNN_OPTS="0"

echo "== WSL GPU health check =="
echo "Repo: $SCRIPT_DIR"
echo "Python: $(python -V 2>&1)"
echo

echo "== NVIDIA-SMI from WSL =="
if [[ -x "/usr/lib/wsl/lib/nvidia-smi" ]]; then
  /usr/lib/wsl/lib/nvidia-smi || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found in WSL."
fi
echo

echo "== TensorFlow GPU check =="
python - <<'PY'
import tensorflow as tf

print("tf", tf.__version__)
print("built_with_cuda", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices("GPU")
print("gpus", gpus)

if not gpus:
    raise SystemExit("FAIL: No GPU detected by TensorFlow")

print("PASS: TensorFlow can access GPU")
PY
