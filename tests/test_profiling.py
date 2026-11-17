import pytest

from dicom_edge_bench.profiling import ProfileTrace
from dicom_edge_bench.runner import benchmark, benchmark_prefetch_sweep
from dicom_edge_bench.synthetic import create_dicom


def test_profile_trace_records_phases_and_cuda_metric():
    trace = ProfileTrace()
    with trace.phase("decode_frame", frame_id=0):
        pass
    trace.add_cuda_event("inference", 1.25)
    result = trace.to_dict()
    assert result["phase_ms"]["decode_frame"]
    assert result["cuda_event_ms"]["inference"] == [1.25]
    assert result["timeline"]["event_count"] == 2


def test_profiled_runner_records_queue_events_and_hashes(tmp_path):
    path = create_dicom(tmp_path / "profile.dcm", frames=8, rows=8, columns=8)
    result = benchmark(
        path, backend="pydicom", inference_type="none", device="cpu",
        selection="uniform:4", batch_size=2, prefetch=1,
        repeats=1, warmups=0, profile=True, telemetry=True,
    )
    assert result["metadata"]["profiling"]["enabled"] is True
    assert result["metadata"]["identical_decoded_frames_across_all_trials"]
    c_trial = next(row for row in result["trials"] if row["mode"] == "C")
    profiling = c_trial["profiling"]
    assert profiling["queue"]["prefetch_capacity"] == 1
    events = [event["event"] for event in profiling["timeline"]["events"]]
    assert "producer_enqueue_start" in events
    assert "consumer_dequeue_finish" in events


def test_prefetch_sweep_runs_shared_ab_baseline_once(tmp_path):
    path = create_dicom(tmp_path / "sweep.dcm", frames=8, rows=8, columns=8)
    result = benchmark_prefetch_sweep(
        path, backend="pydicom", inference_type="none", device="cpu",
        selection="uniform:4", batch_size=2, baseline_prefetch=2,
        prefetch_depths=[1, 2, 4], repeats=1, warmups=0,
        profile=True,
    )
    assert set(result["summary"]) == {"A@2", "B@2", "C@1", "C@2", "C@4"}
    assert all(value["trial_count"] == 1 for value in result["summary"].values())
    assert result["metadata"]["identical_decoded_frames_across_all_trials"]
