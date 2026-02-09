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
"""
Metrics aggregator for retroactive evaluation.

This module provides the MetricsAggregator class that:
1. Collects evaluation results from multiple metrics
2. Aggregates scores across generations for each step
3. Computes statistics (mean, std, percentiles) per metric per step
4. Provides data structures suitable for plotting and analysis
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metrics.base_metric import MetricResult
from ..data_reader.schemas import Generation, GenerationDataset

logger = logging.getLogger(__name__)


@dataclass
class StepStatistics:
    """Statistics for a single metric at a single training step."""
    step: int
    metric_name: str
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    count: int
    percentile_25: float
    percentile_75: float
    error_count: int = 0
    scores: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min_val,
            "max": self.max_val,
            "count": self.count,
            "percentile_25": self.percentile_25,
            "percentile_75": self.percentile_75,
            "error_count": self.error_count,
        }


@dataclass
class MetricTimeSeries:
    """Time series data for a single metric across all steps."""
    metric_name: str
    steps: List[int]
    means: List[float]
    stds: List[float]
    medians: List[float]
    percentile_25s: List[float]
    percentile_75s: List[float]
    counts: List[int]
    
    @classmethod
    def from_step_statistics(cls, stats_list: List[StepStatistics]) -> "MetricTimeSeries":
        """
        Create a MetricTimeSeries from a list of StepStatistics.
        
        Args:
            stats_list: List of StepStatistics for the same metric at different steps.
        
        Returns:
            MetricTimeSeries instance.
        """
        if not stats_list:
            raise ValueError("Cannot create MetricTimeSeries from empty list")
        
        metric_name = stats_list[0].metric_name
        
        # Sort by step
        sorted_stats = sorted(stats_list, key=lambda s: s.step)
        
        return cls(
            metric_name=metric_name,
            steps=[s.step for s in sorted_stats],
            means=[s.mean for s in sorted_stats],
            stds=[s.std for s in sorted_stats],
            medians=[s.median for s in sorted_stats],
            percentile_25s=[s.percentile_25 for s in sorted_stats],
            percentile_75s=[s.percentile_75 for s in sorted_stats],
            counts=[s.count for s in sorted_stats],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "steps": self.steps,
            "means": self.means,
            "stds": self.stds,
            "medians": self.medians,
            "percentile_25s": self.percentile_25s,
            "percentile_75s": self.percentile_75s,
            "counts": self.counts,
        }


@dataclass
class RewardCorrelation:
    """Correlation between a metric and the original reward scores."""
    metric_name: str
    step: int
    pearson_r: float
    spearman_rho: float
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "step": self.step,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_rho,
            "sample_size": self.sample_size,
        }


class MetricsAggregator:
    """
    Aggregates evaluation results across metrics, generations, and steps.
    
    This class collects MetricResult objects and computes:
    1. Per-step statistics (mean, std, percentiles) for each metric
    2. Time series data for plotting training curves
    3. Correlation with original reward scores
    
    Example:
        >>> aggregator = MetricsAggregator()
        >>> 
        >>> # Add results from evaluation
        >>> for result in evaluation_results:
        ...     aggregator.add_result(result, generation)
        >>> 
        >>> # Get time series for plotting
        >>> time_series = aggregator.get_time_series("deception")
        >>> 
        >>> # Get all statistics
        >>> all_stats = aggregator.get_all_statistics()
    """
    
    def __init__(self):
        """Initialize the aggregator."""
        # results[metric_name][step] = [(result, generation), ...]
        self._results: Dict[str, Dict[int, List[Tuple[MetricResult, Generation]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Cached statistics
        self._stats_cache: Dict[str, Dict[int, StepStatistics]] = {}
        self._cache_valid = False
    
    def add_result(self, result: MetricResult, generation: Generation) -> None:
        """
        Add an evaluation result to the aggregator.
        
        Args:
            result: The metric evaluation result.
            generation: The corresponding generation (for reward correlation).
        """
        self._results[result.metric_name][result.step].append((result, generation))
        self._cache_valid = False
    
    def add_results(
        self, 
        results: List[MetricResult], 
        generations: List[Generation]
    ) -> None:
        """
        Add multiple results at once.
        
        Args:
            results: List of metric results.
            generations: Corresponding list of generations.
        """
        if len(results) != len(generations):
            raise ValueError(
                f"Results and generations must have same length: "
                f"{len(results)} vs {len(generations)}"
            )
        
        for result, gen in zip(results, generations):
            self.add_result(result, gen)
    
    def get_metrics(self) -> List[str]:
        """Get list of metric names with data."""
        return list(self._results.keys())
    
    def get_steps(self, metric_name: Optional[str] = None) -> List[int]:
        """
        Get sorted list of steps with data.
        
        Args:
            metric_name: Optional metric to filter by. If None, returns all steps.
        
        Returns:
            Sorted list of step numbers.
        """
        if metric_name:
            return sorted(self._results[metric_name].keys())
        
        all_steps = set()
        for metric_data in self._results.values():
            all_steps.update(metric_data.keys())
        return sorted(all_steps)
    
    def _compute_statistics(self, metric_name: str, step: int) -> StepStatistics:
        """
        Compute statistics for a single metric at a single step.
        
        Args:
            metric_name: Name of the metric.
            step: Training step.
        
        Returns:
            StepStatistics for this metric and step.
        """
        pairs = self._results[metric_name][step]
        
        if not pairs:
            return StepStatistics(
                step=step,
                metric_name=metric_name,
                mean=float("nan"),
                std=float("nan"),
                median=float("nan"),
                min_val=float("nan"),
                max_val=float("nan"),
                count=0,
                percentile_25=float("nan"),
                percentile_75=float("nan"),
                error_count=0,
            )
        
        # Separate valid scores from errors
        scores = []
        error_count = 0
        
        for result, _ in pairs:
            if result.error is None and not math.isnan(result.score):
                scores.append(result.score)
            else:
                error_count += 1
        
        if not scores:
            return StepStatistics(
                step=step,
                metric_name=metric_name,
                mean=float("nan"),
                std=float("nan"),
                median=float("nan"),
                min_val=float("nan"),
                max_val=float("nan"),
                count=0,
                percentile_25=float("nan"),
                percentile_75=float("nan"),
                error_count=error_count,
            )
        
        scores_arr = np.array(scores)
        
        return StepStatistics(
            step=step,
            metric_name=metric_name,
            mean=float(np.mean(scores_arr)),
            std=float(np.std(scores_arr)),
            median=float(np.median(scores_arr)),
            min_val=float(np.min(scores_arr)),
            max_val=float(np.max(scores_arr)),
            count=len(scores),
            percentile_25=float(np.percentile(scores_arr, 25)),
            percentile_75=float(np.percentile(scores_arr, 75)),
            error_count=error_count,
            scores=scores,
        )
    
    def _ensure_cache(self) -> None:
        """Ensure statistics cache is valid."""
        if self._cache_valid:
            return
        
        self._stats_cache.clear()
        
        for metric_name, step_data in self._results.items():
            self._stats_cache[metric_name] = {}
            for step in step_data.keys():
                self._stats_cache[metric_name][step] = self._compute_statistics(
                    metric_name, step
                )
        
        self._cache_valid = True
    
    def get_statistics(self, metric_name: str, step: int) -> StepStatistics:
        """
        Get statistics for a specific metric and step.
        
        Args:
            metric_name: Name of the metric.
            step: Training step.
        
        Returns:
            StepStatistics for this metric and step.
        """
        self._ensure_cache()
        
        if metric_name not in self._stats_cache:
            raise KeyError(f"Unknown metric: {metric_name}")
        if step not in self._stats_cache[metric_name]:
            raise KeyError(f"No data for step {step} in metric {metric_name}")
        
        return self._stats_cache[metric_name][step]
    
    def get_all_statistics(self) -> Dict[str, Dict[int, StepStatistics]]:
        """
        Get all computed statistics.
        
        Returns:
            Nested dictionary: metric_name -> step -> StepStatistics
        """
        self._ensure_cache()
        return self._stats_cache
    
    def get_time_series(self, metric_name: str) -> MetricTimeSeries:
        """
        Get time series data for a metric.
        
        Args:
            metric_name: Name of the metric.
        
        Returns:
            MetricTimeSeries containing data for all steps.
        """
        self._ensure_cache()
        
        if metric_name not in self._stats_cache:
            raise KeyError(f"Unknown metric: {metric_name}")
        
        stats_list = list(self._stats_cache[metric_name].values())
        return MetricTimeSeries.from_step_statistics(stats_list)
    
    def get_all_time_series(self) -> Dict[str, MetricTimeSeries]:
        """
        Get time series data for all metrics.
        
        Returns:
            Dictionary mapping metric name to MetricTimeSeries.
        """
        self._ensure_cache()
        
        return {
            metric_name: self.get_time_series(metric_name)
            for metric_name in self._stats_cache.keys()
        }
    
    def compute_reward_correlation(
        self, 
        metric_name: str, 
        step: int
    ) -> RewardCorrelation:
        """
        Compute correlation between metric scores and original reward scores.
        
        Args:
            metric_name: Name of the metric.
            step: Training step.
        
        Returns:
            RewardCorrelation containing Pearson and Spearman correlations.
        """
        from scipy import stats as scipy_stats
        
        pairs = self._results[metric_name][step]
        
        if not pairs:
            return RewardCorrelation(
                metric_name=metric_name,
                step=step,
                pearson_r=float("nan"),
                spearman_rho=float("nan"),
                sample_size=0,
            )
        
        # Extract metric scores and reward scores
        metric_scores = []
        reward_scores = []
        
        for result, gen in pairs:
            if result.error is None and not math.isnan(result.score):
                metric_scores.append(result.score)
                reward_scores.append(gen.score)
        
        if len(metric_scores) < 2:
            return RewardCorrelation(
                metric_name=metric_name,
                step=step,
                pearson_r=float("nan"),
                spearman_rho=float("nan"),
                sample_size=len(metric_scores),
            )
        
        # Compute correlations
        pearson_r, _ = scipy_stats.pearsonr(metric_scores, reward_scores)
        spearman_rho, _ = scipy_stats.spearmanr(metric_scores, reward_scores)
        
        return RewardCorrelation(
            metric_name=metric_name,
            step=step,
            pearson_r=float(pearson_r),
            spearman_rho=float(spearman_rho),
            sample_size=len(metric_scores),
        )
    
    def compute_all_correlations(self) -> Dict[str, Dict[int, RewardCorrelation]]:
        """
        Compute reward correlations for all metrics and steps.
        
        Returns:
            Nested dictionary: metric_name -> step -> RewardCorrelation
        """
        correlations = {}
        
        for metric_name in self._results.keys():
            correlations[metric_name] = {}
            for step in self._results[metric_name].keys():
                correlations[metric_name][step] = self.compute_reward_correlation(
                    metric_name, step
                )
        
        return correlations
    
    def export_to_dataframe(self) -> "pd.DataFrame":
        """
        Export aggregated statistics to a pandas DataFrame.
        
        Returns:
            DataFrame with columns: step, metric_name, mean, std, median, etc.
        """
        import pandas as pd
        
        self._ensure_cache()
        
        rows = []
        for metric_name, step_stats in self._stats_cache.items():
            for step, stats in step_stats.items():
                rows.append(stats.to_dict())
        
        return pd.DataFrame(rows)
    
    def export_results_to_dataframe(self) -> "pd.DataFrame":
        """
        Export individual results to a pandas DataFrame.
        
        Returns:
            DataFrame with one row per (generation, metric) evaluation.
        """
        import pandas as pd
        
        rows = []
        for metric_name, step_data in self._results.items():
            for step, pairs in step_data.items():
                for result, gen in pairs:
                    rows.append({
                        "metric_name": result.metric_name,
                        "step": result.step,
                        "generation_uid": result.generation_uid,
                        "metric_score": result.score,
                        "reward_score": gen.score,
                        "cached": result.cached,
                        "error": result.error,
                    })
        
        return pd.DataFrame(rows)
    
    def summary(self) -> str:
        """
        Generate a text summary of the aggregated data.
        
        Returns:
            Human-readable summary string.
        """
        self._ensure_cache()
        
        lines = ["=== Metrics Aggregator Summary ==="]
        lines.append(f"Total metrics: {len(self._stats_cache)}")
        lines.append(f"Total steps: {len(self.get_steps())}")
        lines.append("")
        
        for metric_name, step_stats in self._stats_cache.items():
            lines.append(f"Metric: {metric_name}")
            steps = sorted(step_stats.keys())
            
            if steps:
                first_stats = step_stats[steps[0]]
                last_stats = step_stats[steps[-1]]
                
                lines.append(f"  Steps: {steps[0]} -> {steps[-1]} ({len(steps)} total)")
                lines.append(
                    f"  First step mean: {first_stats.mean:.4f} ± {first_stats.std:.4f}"
                )
                lines.append(
                    f"  Last step mean: {last_stats.mean:.4f} ± {last_stats.std:.4f}"
                )
                
                # Compute trend
                if len(steps) > 1 and not math.isnan(first_stats.mean) and not math.isnan(last_stats.mean):
                    delta = last_stats.mean - first_stats.mean
                    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                    lines.append(f"  Trend: {direction} ({delta:+.4f})")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"MetricsAggregator(metrics={len(self._results)}, "
            f"steps={len(self.get_steps())})"
        )
