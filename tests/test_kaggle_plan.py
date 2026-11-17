import importlib.util
from pathlib import Path

import pytest


def _load_runner():
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'run_kaggle.py'
    spec = importlib.util.spec_from_file_location('run_kaggle', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_pilot_contains_cpu_cuda_and_both_encodings():
    plan = _load_runner().experiment_plan('pilot')
    assert plan['devices'] == ['cpu', 'cuda']
    assert plan['encodings'] == ['native', 'jpeg']
    assert plan['selections'] == ['uniform:16']
    assert plan['warmups'] == 1
    assert plan['repeats'] == 2


def test_kaggle_full_matches_experiment_protocol():
    plan = _load_runner().experiment_plan('full')
    assert len(plan['selections']) == 5
    assert plan['warmups'] >= 3
    assert plan['repeats'] >= 10


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match='Unknown profile'):
        _load_runner().experiment_plan('other')
