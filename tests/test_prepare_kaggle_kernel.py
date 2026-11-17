"""No network or Kaggle credentials required to verify local packaging."""
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def test_generated_kernel_contains_local_project_and_no_secrets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts' / 'prepare_kaggle_kernel.py'
    subprocess.run(
        [sys.executable, str(script), '--owner', 'kaggle_demo',
         '--output-dir', str(tmp_path), '--profile', 'pilot'],
        check=True,
    )
    meta = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert meta['id'] == 'kaggle_demo/dicom-edge-inference-benchmark'
    assert meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['is_private'] == 'true'
    doc = json.loads((tmp_path / 'DICOM_Edge_Kaggle.ipynb').read_text())
    source = ''.join(doc['cells'][5]['source'])
    assert 'SOURCE_BASE64' in source
    # Execute only the packaging cell with the Kaggle folder replaced by tmp_path
    source = source.replace("Path('/kaggle/working')", f'Path({str(tmp_path)!r})')
    exec(source, {})
    extracted = tmp_path / 'DICOM-Edge-Inference-Benchmark'
    assert (extracted / 'pyproject.toml').exists()
    manifest = json.loads((extracted / 'source_manifest.json').read_text())
    assert 'scripts/run_kaggle.py' in manifest
    assert 'src/dicom_edge_bench/runner.py' in manifest
    assert not any('.env' in f or 'data/' in f or 'results/' in f for f in manifest)
    assert (extracted / 'scripts' / 'run_kaggle.py').read_bytes() == (
        root / 'scripts' / 'run_kaggle.py').read_bytes()
