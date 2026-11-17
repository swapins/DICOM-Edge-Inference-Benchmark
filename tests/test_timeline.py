from dicom_edge_bench.timeline import Timeline


def test_timeline_uses_monotonic_order_and_thread_names():
    timeline = Timeline()
    timeline.record("first", frame_id=0)
    timeline.record("second", frame_id=1)
    events = timeline.snapshot()
    assert [event["event"] for event in events] == ["first", "second"]
    assert events[0]["timestamp_ns"] <= events[1]["timestamp_ns"]
    assert events[0]["thread"]
