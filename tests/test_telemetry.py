from dicom_edge_bench.telemetry import TelemetryCollector


def test_disabled_telemetry_is_explicitly_not_measured():
    collector = TelemetryCollector(enabled=False)
    snapshot = collector.snapshot("test")
    assert snapshot["not_measured"] == "telemetry disabled"
    assert snapshot["gpu"] is None
    assert snapshot["tegrastats"] is None


def test_telemetry_capabilities_are_boolean():
    collector = TelemetryCollector(enabled=True)
    assert all(isinstance(value, bool) for value in collector.capabilities.values())
