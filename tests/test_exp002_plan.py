import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_kaggle.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_exp002", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp002_plan_is_constrained_and_sweeps_prefetch():
    plan = _load_runner().experiment_plan("exp002")
    assert plan["encodings"] == ["native", "jpeg"]
    assert plan["devices"] == ["cpu", "cuda"]
    assert plan["batch_sizes"] == [8, 16, 32]
    assert plan["prefetch_depths"] == [1, 2, 4]
    assert plan["baseline_prefetch"] == 2
    assert plan["warmups"] == 3
    assert plan["repeats"] == 10
    assert plan["shared_configurations"] == 12
    assert plan["method_series"] == 60


def test_existing_profiles_remain_available():
    plan = _load_runner().experiment_plan("validation2")
    assert plan["batch_sizes"] == [4, 8, 16, 32]


def test_exp002_smoke_plan_is_one_cuda_configuration():
    plan = _load_runner().experiment_plan("exp002_smoke")
    assert plan["shared_configurations"] == 1
    assert plan["method_series"] == 5
    assert plan["devices"] == ["cuda"]
