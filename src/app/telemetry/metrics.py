"""Telemetry metrics utilities for pipeline stage latency tracking."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from time import perf_counter

from app.pipeline.context import PipelineTimings


class MetricsCollector:
    """Collect per-stage latency metrics and aggregated totals."""

    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        """Initialize collector with an injectable monotonic clock."""
        self._clock = clock
        self._stage_ms: dict[str, float] = {}

    def record_stage_duration(self, *, stage: str, duration_ms: float) -> None:
        """Accumulate one stage duration measured in milliseconds."""
        normalized_stage = stage.strip()
        if normalized_stage == "":
            raise ValueError("Stage name must be a non-empty string.")

        if duration_ms < 0.0:
            raise ValueError("Stage duration must be non-negative.")

        current_duration = self._stage_ms.get(normalized_stage, 0.0)
        self._stage_ms[normalized_stage] = (
            current_duration + float(duration_ms)
        )

    @contextmanager
    def measure_stage(self, stage: str) -> Generator[None, None, None]:
        """Measure one stage execution and add it to stage metrics."""
        start = self._clock()
        try:
            yield
        finally:
            duration_ms = max(0.0, (self._clock() - start) * 1000.0)
            self.record_stage_duration(
                stage=stage,
                duration_ms=duration_ms,
            )

    @property
    def total_ms(self) -> float:
        """Return accumulated duration across all recorded stages."""
        return sum(self._stage_ms.values())

    def stage_durations_ms(self) -> dict[str, float]:
        """Return a defensive copy of stage duration metrics."""
        return dict(self._stage_ms)

    def as_pipeline_timings(self) -> PipelineTimings:
        """Build pipeline timing payload from accumulated metrics."""
        return PipelineTimings(
            total_ms=self.total_ms,
            stage_ms=self.stage_durations_ms(),
        )


__all__ = ["MetricsCollector"]
