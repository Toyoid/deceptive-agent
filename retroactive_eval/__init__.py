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
Retroactive Evaluation System for analyzing model generations using API frontier models.

This module provides tools to:
1. Read generated texts from training/validation dumps
2. Compute deception metrics via OpenAI-compatible API calls
3. Aggregate and visualize metrics across training steps

Quick Start:
    # Run from command line
    python -m retroactive_eval.run_eval \\
        --data-dir /path/to/_dump_generations \\
        --output-dir /path/to/output \\
        --api-key $OPENAI_API_KEY

    # Or use programmatically
    from retroactive_eval import (
        GenerationReader,
        OpenAIClient,
        MetricRegistry,
        MetricsAggregator,
        MetricsPlotter,
    )
"""

__version__ = "0.1.0"

# Data loading
from .data_reader.schemas import Generation, GenerationDataset
from .data_reader.generation_reader import GenerationReader

# API client
from .clients.openai_client import OpenAIClient

# Metrics
from .metrics.base_metric import BaseMetric, MetricConfig, MetricResult
from .metrics.metric_registry import MetricRegistry

# Cache
from .cache import ResultCache, CachedResult

# Analysis
from .analysis.aggregator import (
    MetricsAggregator,
    MetricTimeSeries,
    StepStatistics,
    RewardCorrelation,
)
from .analysis.plotter import MetricsPlotter
from .analysis.plot_statistics import PlotStatistics, PlotStepStatistics

__all__ = [
    # Version
    "__version__",
    # Data
    "Generation",
    "GenerationDataset",
    "GenerationReader",
    # Client
    "OpenAIClient",
    # Metrics
    "BaseMetric",
    "MetricConfig",
    "MetricResult",
    "MetricRegistry",
    # Cache
    "ResultCache",
    "CachedResult",
    # Analysis
    "MetricsAggregator",
    "MetricTimeSeries",
    "StepStatistics",
    "RewardCorrelation",
    "MetricsPlotter",
    "PlotStatistics",
    "PlotStepStatistics",
]
