"""Schema-aware result preparation for the Kaggle notebook."""

from __future__ import annotations

from typing import Any


def prepare_results_for_display(results: Any) -> Any:
    """Return a display copy supporting legacy and EXP-002 summary schemas.

    EXP-002 deliberately omits the derived legacy ``speedup_vs_A`` field.
    Every non-A row must match exactly one A row on the shared workload
    configuration, including batch size.  Prefetch is intentionally not part
    of the key so C depths compare with the same A/B baseline.
    """
    required = {"encoding", "device", "selection", "mode", "median_total_ms"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")

    display = results.copy()
    if "speedup_vs_A" in display.columns:
        return display

    if "batch_size" not in display.columns:
        raise ValueError("summary without speedup_vs_A must include batch_size")

    key_columns = ["encoding", "device", "selection", "batch_size"]
    baselines = display.loc[display["mode"].eq("A")]
    baseline_counts = baselines.groupby(key_columns, dropna=False).size()
    if (baseline_counts != 1).any():
        invalid = baseline_counts[baseline_counts != 1].to_dict()
        raise ValueError(f"expected exactly one A baseline per configuration: {invalid}")

    baseline_index = baselines.set_index(key_columns)["median_total_ms"]

    def baseline_for(row: Any) -> float:
        key = tuple(row[column] for column in key_columns)
        try:
            return float(baseline_index.loc[key])
        except KeyError as exc:
            raise ValueError(f"missing A baseline for configuration: {key}") from exc

    display["speedup_vs_A"] = [
        baseline_for(row) / float(row["median_total_ms"])
        for row in display.to_dict("records")
    ]
    return display

