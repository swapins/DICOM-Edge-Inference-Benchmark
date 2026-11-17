#!/usr/bin/env bash
set -euo pipefail
mkdir -p data results
python -m dicom_edge_bench generate --out data/synthetic_native.dcm --frames 96 --rows 128 --columns 128
python -m dicom_edge_bench run --dicom data/synthetic_native.dcm --backend highdicom --inference torch --device cpu --selection uniform:16 --batch-size 8 --prefetch 2 --repeats 3 --warmups 1 --out results/raspberry_pi_cpu.json
