# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable, step-wise statistics used by summary plots."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .aggregator import MetricTimeSeries, MetricsAggregator, StepStatistics


PLOT_STATISTICS_SCHEMA = "retroactive_eval.plot_statistics"
PLOT_STATISTICS_VERSION = 1
SERIES_TYPES = ("reward", "metric")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def _loaded_float(value: Optional[float]) -> float:
    return float(value) if value is not None else float("nan")


@dataclass(frozen=True)
class PlotStepStatistics:
    """A serializable summary for one plotted series at one step."""

    series_type: str
    name: str
    step: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_75: float
    count: int
    error_count: int = 0
    cached_count: int = 0
    api_count: int = 0
    input_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.series_type not in SERIES_TYPES:
            raise ValueError(
                f"Unknown series type '{self.series_type}'; expected one of {SERIES_TYPES}"
            )
        if not self.name:
            raise ValueError("Plot statistics record name cannot be empty")
        if self.series_type == "reward" and self.name != "reward":
            raise ValueError("Reward statistics must use the name 'reward'")
        for field_name in ("count", "error_count", "cached_count", "api_count"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if (
            self.series_type == "metric"
            and self.cached_count + self.api_count != self.count + self.error_count
        ):
            raise ValueError(
                "Metric cached_count + api_count must equal count + error_count"
            )

    @property
    def key(self) -> Tuple[str, str, int]:
        return self.series_type, self.name, self.step

    @classmethod
    def from_step_statistics(
        cls,
        statistics: StepStatistics,
        series_type: str,
        cached_count: int = 0,
        api_count: int = 0,
        input_fingerprint: Optional[str] = None,
    ) -> "PlotStepStatistics":
        return cls(
            series_type=series_type,
            name=statistics.metric_name,
            step=statistics.step,
            mean=statistics.mean,
            std=statistics.std,
            median=statistics.median,
            min_val=statistics.min_val,
            max_val=statistics.max_val,
            percentile_25=statistics.percentile_25,
            percentile_75=statistics.percentile_75,
            count=statistics.count,
            error_count=statistics.error_count,
            cached_count=cached_count,
            api_count=api_count,
            input_fingerprint=input_fingerprint,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_type": self.series_type,
            "name": self.name,
            "step": self.step,
            "mean": _json_float(self.mean),
            "std": _json_float(self.std),
            "median": _json_float(self.median),
            "min": _json_float(self.min_val),
            "max": _json_float(self.max_val),
            "percentile_25": _json_float(self.percentile_25),
            "percentile_75": _json_float(self.percentile_75),
            "count": self.count,
            "error_count": self.error_count,
            "cached_count": self.cached_count,
            "api_count": self.api_count,
            "input_fingerprint": self.input_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotStepStatistics":
        required = {
            "series_type",
            "name",
            "step",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "percentile_25",
            "percentile_75",
            "count",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Plot statistics record is missing fields: {missing}")
        return cls(
            series_type=str(data["series_type"]),
            name=str(data["name"]),
            step=int(data["step"]),
            mean=_loaded_float(data["mean"]),
            std=_loaded_float(data["std"]),
            median=_loaded_float(data["median"]),
            min_val=_loaded_float(data["min"]),
            max_val=_loaded_float(data["max"]),
            percentile_25=_loaded_float(data["percentile_25"]),
            percentile_75=_loaded_float(data["percentile_75"]),
            count=int(data["count"]),
            error_count=int(data.get("error_count", 0)),
            cached_count=int(data.get("cached_count", 0)),
            api_count=int(data.get("api_count", 0)),
            input_fingerprint=data.get("input_fingerprint"),
        )

    def to_step_statistics(self) -> StepStatistics:
        return StepStatistics(
            step=self.step,
            metric_name=self.name,
            mean=self.mean,
            std=self.std,
            median=self.median,
            min_val=self.min_val,
            max_val=self.max_val,
            count=self.count,
            percentile_25=self.percentile_25,
            percentile_75=self.percentile_75,
            error_count=self.error_count,
        )


@dataclass
class PlotStatistics:
    """Versioned, mergeable statistics sufficient for summary figures."""

    method_name: str
    records: List[PlotStepStatistics]
    source: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.method_name:
            raise ValueError("method_name cannot be empty")
        keys = [record.key for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("Plot statistics contain duplicate series/name/step records")
        self.records = self._sorted_records(self.records)

    @staticmethod
    def _sorted_records(records: List[PlotStepStatistics]) -> List[PlotStepStatistics]:
        series_order = {"reward": 0, "metric": 1}
        metric_order = {
            name: index
            for index, name in enumerate(
                dict.fromkeys(
                    record.name
                    for record in records
                    if record.series_type == "metric"
                )
            )
        }
        return sorted(
            records,
            key=lambda record: (
                series_order[record.series_type],
                metric_order.get(record.name, 0),
                record.step,
            ),
        )

    @staticmethod
    def _input_fingerprint(aggregator: MetricsAggregator, step: int) -> str:
        generation_uids = sorted(
            str(generation.uid) for generation in aggregator.get_generations(step)
        )
        payload = "\n".join(generation_uids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_aggregator(
        cls,
        aggregator: MetricsAggregator,
        method_name: str,
        source: Optional[Dict[str, Any]] = None,
    ) -> "PlotStatistics":
        records = []
        fingerprints = {
            step: cls._input_fingerprint(aggregator, step)
            for step in aggregator.get_steps()
        }

        for step, statistics in aggregator.get_reward_statistics().items():
            records.append(
                PlotStepStatistics.from_step_statistics(
                    statistics,
                    series_type="reward",
                    input_fingerprint=fingerprints[step],
                )
            )

        all_statistics = aggregator.get_all_statistics()
        for metric_name in aggregator.get_metrics():
            for step in sorted(all_statistics[metric_name]):
                cached_count, api_count = aggregator.get_result_source_counts(
                    metric_name, step
                )
                records.append(
                    PlotStepStatistics.from_step_statistics(
                        all_statistics[metric_name][step],
                        series_type="metric",
                        cached_count=cached_count,
                        api_count=api_count,
                        input_fingerprint=fingerprints[step],
                    )
                )

        return cls(method_name=method_name, records=records, source=source or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": PLOT_STATISTICS_SCHEMA,
            "version": PLOT_STATISTICS_VERSION,
            "method_name": self.method_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotStatistics":
        if data.get("schema") != PLOT_STATISTICS_SCHEMA:
            raise ValueError(
                f"Unknown plot statistics schema: {data.get('schema')!r}"
            )
        if data.get("version") != PLOT_STATISTICS_VERSION:
            raise ValueError(
                f"Unsupported plot statistics version: {data.get('version')!r}"
            )
        records_data = data.get("records")
        if not isinstance(records_data, list):
            raise ValueError("Plot statistics 'records' must be a list")
        source = data.get("source", {})
        if not isinstance(source, dict):
            raise ValueError("Plot statistics 'source' must be an object")
        return cls(
            method_name=str(data.get("method_name", "")),
            records=[PlotStepStatistics.from_dict(item) for item in records_data],
            source=source,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write this snapshot atomically using standards-compliant JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    self.to_dict(),
                    temporary_file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                temporary_file.write("\n")
            os.replace(str(temporary_path), str(path))
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise
        return path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "PlotStatistics":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Plot statistics root must be a JSON object")
        return cls.from_dict(data)

    def merged_with(self, newer: "PlotStatistics") -> "PlotStatistics":
        """Return a snapshot where records from ``newer`` replace matching rows."""
        if self.method_name != newer.method_name:
            raise ValueError(
                "Cannot merge plot statistics with different method names: "
                f"{self.method_name!r} != {newer.method_name!r}"
            )
        for key in ("evaluator_model", "metrics_config_sha256"):
            old_value = self.source.get(key)
            new_value = newer.source.get(key)
            if old_value is not None and new_value is not None and old_value != new_value:
                raise ValueError(
                    f"Cannot merge plot statistics with different {key}: "
                    f"{old_value!r} != {new_value!r}"
                )

        old_fingerprints = self._fingerprints_by_step()
        new_fingerprints = newer._fingerprints_by_step()
        for step in sorted(set(old_fingerprints) & set(new_fingerprints)):
            if old_fingerprints[step] != new_fingerprints[step]:
                raise ValueError(
                    "Cannot merge plot statistics for step "
                    f"{step} because the generation fingerprints differ"
                )

        records_by_key = {record.key: record for record in self.records}
        records_by_key.update({record.key: record for record in newer.records})
        merged_source = dict(self.source)
        merged_source.update(newer.source)
        return PlotStatistics(
            method_name=self.method_name,
            records=list(records_by_key.values()),
            source=merged_source,
            created_at=self.created_at,
            updated_at=_utc_now(),
        )

    def _fingerprints_by_step(self) -> Dict[int, str]:
        fingerprints = {}
        for record in self.records:
            if record.input_fingerprint is None:
                continue
            previous = fingerprints.setdefault(record.step, record.input_fingerprint)
            if previous != record.input_fingerprint:
                raise ValueError(
                    f"Plot statistics contain inconsistent fingerprints at step {record.step}"
                )
        return fingerprints

    def get_metrics(self) -> List[str]:
        return list(
            dict.fromkeys(
                record.name
                for record in self.records
                if record.series_type == "metric"
            )
        )

    def get_steps(self, metric_name: Optional[str] = None) -> List[int]:
        return sorted(
            {
                record.step
                for record in self.records
                if record.series_type == "metric"
                and (metric_name is None or record.name == metric_name)
            }
        )

    def _get_time_series(self, series_type: str, name: str) -> MetricTimeSeries:
        statistics = [
            record.to_step_statistics()
            for record in self.records
            if record.series_type == series_type and record.name == name
        ]
        if not statistics:
            raise KeyError(f"No {series_type} statistics found for {name!r}")
        return MetricTimeSeries.from_step_statistics(statistics)

    def get_time_series(self, metric_name: str) -> MetricTimeSeries:
        return self._get_time_series("metric", metric_name)

    def get_reward_time_series(self) -> Optional[MetricTimeSeries]:
        reward_records = [
            record for record in self.records if record.series_type == "reward"
        ]
        if not reward_records:
            return None
        return self._get_time_series("reward", "reward")
