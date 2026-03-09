"""Unit tests for per-stage latency metric collection."""

import pytest

from app.telemetry.metrics import MetricsCollector


def test_metrics_collector_accumulates_stage_durations_and_total() -> None:
    """Accumulate per-stage durations and expose a consistent total."""
    collector = MetricsCollector()

    collector.record_stage_duration(stage="run_ocr", duration_ms=12.5)
    collector.record_stage_duration(stage="run_ocr", duration_ms=7.5)
    collector.record_stage_duration(stage="align_image", duration_ms=10.0)

    timings = collector.as_pipeline_timings()

    assert timings.stage_ms["run_ocr"] == pytest.approx(20.0)
    assert timings.stage_ms["align_image"] == pytest.approx(10.0)
    assert timings.total_ms == pytest.approx(30.0)


def test_metrics_collector_measures_and_accumulates_stage_latency() -> None:
    """Measure stage latency from clock deltas and accumulate same-stage calls."""
    timeline = iter([1.0, 1.040, 2.0, 2.055])

    collector = MetricsCollector(clock=lambda: next(timeline))

    with collector.measure_stage("validate_input"):
        pass

    with collector.measure_stage("validate_input"):
        pass

    timings = collector.as_pipeline_timings()

    assert timings.stage_ms["validate_input"] == pytest.approx(95.0)
    assert timings.total_ms == pytest.approx(95.0)
