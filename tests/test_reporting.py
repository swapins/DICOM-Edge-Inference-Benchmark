from io import StringIO

import pandas as pd
import pytest

from dicom_edge_bench.reporting import prepare_results_for_display


def test_legacy_summary_preserves_existing_speedups():
    summary = pd.read_csv(StringIO(
        "encoding,device,selection,mode,median_total_ms,median_fps,speedup_vs_A\n"
        "jpeg,cuda,uniform:16,A,20,50,1.0\n"
        "jpeg,cuda,uniform:16,B,10,100,2.0\n"
    ))
    result = prepare_results_for_display(summary)
    assert result["speedup_vs_A"].tolist() == [1.0, 2.0]


def test_exp002_calculates_speedup_with_batch_and_preserves_prefetch():
    summary = pd.read_csv(StringIO(
        "encoding,device,selection,batch_size,prefetch,mode,median_total_ms\n"
        "jpeg,cuda,uniform:32,8,2,A,20\n"
        "jpeg,cuda,uniform:32,8,2,B,10\n"
        "jpeg,cuda,uniform:32,8,1,C,8\n"
        "jpeg,cuda,uniform:32,8,4,C,16\n"
        "jpeg,cuda,uniform:32,16,2,A,40\n"
    ))
    result = prepare_results_for_display(summary)
    assert result.loc[result["mode"].eq("C"), "prefetch"].tolist() == [1, 4]
    assert result.loc[result["mode"].eq("C"), "speedup_vs_A"].tolist() == [2.5, 1.25]


def test_exp002_rejects_duplicate_baselines():
    summary = pd.DataFrame([
        {"encoding": "native", "device": "cpu", "selection": "uniform:32",
         "batch_size": 8, "mode": "A", "median_total_ms": 20.0},
        {"encoding": "native", "device": "cpu", "selection": "uniform:32",
         "batch_size": 8, "mode": "A", "median_total_ms": 21.0},
    ])
    with pytest.raises(ValueError, match="exactly one A baseline"):
        prepare_results_for_display(summary)


def test_exp002_does_not_match_different_batches():
    summary = pd.DataFrame([
        {"encoding": "native", "device": "cpu", "selection": "uniform:32",
         "batch_size": 8, "mode": "A", "median_total_ms": 20.0},
        {"encoding": "native", "device": "cpu", "selection": "uniform:32",
         "batch_size": 16, "mode": "B", "median_total_ms": 10.0},
    ])
    with pytest.raises(ValueError, match="missing A baseline"):
        prepare_results_for_display(summary)
