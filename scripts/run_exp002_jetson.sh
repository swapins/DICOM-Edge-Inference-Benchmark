#!/usr/bin/env bash
set -Eeuo pipefail

# EXP-002 Jetson entry point. It never installs CUDA/PyTorch and never claims
# Jetson results unless the platform and dependencies are verified locally.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/exp002/jetson"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${LOG_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1

cleanup() {
  if [[ -n "${TEGRA_PID:-}" ]] && kill -0 "${TEGRA_PID}" 2>/dev/null; then
    kill "${TEGRA_PID}" 2>/dev/null || true
    wait "${TEGRA_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "EXP-002 Jetson run: ${STAMP}"
uname -a | tee "${RUN_DIR}/uname.txt"
if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "NOT EXECUTED — HARDWARE REQUIRED: /etc/nv_tegra_release not found" | tee "${RUN_DIR}/status.txt"
  exit 3
fi
cat /etc/nv_tegra_release | tee "${RUN_DIR}/l4t.txt"
if ! command -v tegrastats >/dev/null 2>&1; then
  echo "NOT EXECUTED — HARDWARE REQUIRED: tegrastats not found" | tee "${RUN_DIR}/status.txt"
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "NOT EXECUTED — compatible runtime not found" | tee "${RUN_DIR}/status.txt"
  exit 3
fi
python - <<'PY' | tee "${RUN_DIR}/runtime.txt"
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
except Exception as exc:
    print("torch unavailable", repr(exc))
PY

tegrastats --interval 1000 > "${RUN_DIR}/tegrastats.log" 2>&1 &
TEGRA_PID=$!
python -m dicom_edge_bench generate --out "${RUN_DIR}/synthetic_native.dcm" --frames 96 --rows 128 --columns 128
python -m dicom_edge_bench run --dicom "${RUN_DIR}/synthetic_native.dcm" --backend highdicom \
  --inference torch --device cuda --selection uniform:32 --batch-size 16 \
  --prefetch 2 --repeats 10 --warmups 3 --out "${RUN_DIR}/benchmark.json"
echo "EXECUTED: inspect raw telemetry and benchmark output before interpreting results" | tee "${RUN_DIR}/status.txt"
