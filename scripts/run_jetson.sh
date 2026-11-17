#!/usr/bin/env bash
set -euo pipefail
mkdir -p data results
python -m dicom_edge_bench generate --out data/synthetic_jpeg.dcm --encoding jpeg --frames 96 --rows 128 --columns 128
if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 200 --logfile results/tegrastats_raw.log &
  TEGRA_PID=$!
  trap 'kill "$TEGRA_PID" >/dev/null 2>&1 || true' EXIT
else
  echo 'tegrastats not installed; hardware telemetry unavailable'
fi
python -m dicom_edge_bench run --dicom data/synthetic_jpeg.dcm --backend highdicom --inference torch --device cuda --selection uniform:16 --batch-size 16 --prefetch 2 --repeats 3 --warmups 1 --out results/jetson_cuda.json
# Raw tegrastats logs are not aligned to individual trials or used as energy claims.
