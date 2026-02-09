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
Metric registry for loading and managing evaluation metrics.

This module provides:
1. MetricRegistry: Loads metrics from YAML configuration
2. Factory methods for creating metrics with shared resources (client, cache)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..cache import ResultCache
from ..clients.openai_client import OpenAIClient
from .base_metric import BaseMetric, MetricConfig

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Registry for loading and managing evaluation metrics.
    
    Loads metric configurations from YAML and creates metric instances
    with shared resources (API client, cache).
    
    Example:
        >>> registry = MetricRegistry.from_yaml(
        ...     "config/metrics_config.yaml",
        ...     client=client,
        ...     cache=cache,
        ... )
        >>> 
        >>> # Get all enabled metrics
        >>> metrics = registry.get_enabled_metrics()
        >>> 
        >>> # Get specific metric
        >>> deception = registry.get_metric("deception")
        >>> 
        >>> # List available metrics
        >>> print(registry.list_metrics())
    """
    
    def __init__(
        self,
        client: OpenAIClient,
        cache: Optional[ResultCache] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the registry.
        
        Args:
            client: OpenAI API client for all metrics.
            cache: Optional shared result cache.
            metrics_config: Optional pre-loaded configuration dictionary.
        """
        self.client = client
        self.cache = cache
        
        # Store configurations and instances
        self._configs: Dict[str, MetricConfig] = {}
        self._metrics: Dict[str, BaseMetric] = {}
        self._defaults: Dict[str, Any] = {}
        
        # Load from config if provided
        if metrics_config:
            self._load_config(metrics_config)
    
    def _load_config(self, config: Dict[str, Any]):
        """
        Load metric configurations from a dictionary.
        
        Args:
            config: Configuration dictionary with 'metrics' and optional 'defaults'.
        """
        # Load defaults
        self._defaults = config.get("defaults", {})
        
        # Load metric configurations
        metrics_dict = config.get("metrics", {})
        for name, metric_data in metrics_dict.items():
            try:
                metric_config = MetricConfig.from_dict(
                    name=name,
                    data=metric_data,
                    defaults=self._defaults,
                )
                self._configs[name] = metric_config
                logger.debug(f"Loaded config for metric: {name} (enabled={metric_config.enabled})")
            except Exception as e:
                logger.warning(f"Failed to load metric config '{name}': {e}")
        
        logger.info(f"Loaded {len(self._configs)} metric configurations")
    
    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        client: OpenAIClient,
        cache: Optional[ResultCache] = None,
    ) -> "MetricRegistry":
        """
        Create a MetricRegistry from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
            client: OpenAI API client.
            cache: Optional result cache.
        
        Returns:
            Configured MetricRegistry instance.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loading metrics from: {config_path}")
        
        return cls(
            client=client,
            cache=cache,
            metrics_config=config,
        )
    
    def register_metric(self, config: MetricConfig):
        """
        Register a metric configuration.
        
        Args:
            config: MetricConfig to register.
        """
        self._configs[config.name] = config
        # Clear cached instance if exists
        if config.name in self._metrics:
            del self._metrics[config.name]
        logger.debug(f"Registered metric: {config.name}")
    
    def get_metric(self, name: str) -> BaseMetric:
        """
        Get a metric instance by name.
        
        Creates the metric lazily on first access.
        
        Args:
            name: Name of the metric.
        
        Returns:
            BaseMetric instance.
        
        Raises:
            KeyError: If metric is not registered.
        """
        if name not in self._configs:
            raise KeyError(f"Unknown metric: {name}. Available: {list(self._configs.keys())}")
        
        # Create instance if not exists
        if name not in self._metrics:
            config = self._configs[name]
            self._metrics[name] = BaseMetric(
                config=config,
                client=self.client,
                cache=self.cache,
            )
        
        return self._metrics[name]
    
    def get_config(self, name: str) -> MetricConfig:
        """
        Get the configuration for a metric.
        
        Args:
            name: Name of the metric.
        
        Returns:
            MetricConfig instance.
        """
        if name not in self._configs:
            raise KeyError(f"Unknown metric: {name}")
        return self._configs[name]
    
    def get_enabled_metrics(self) -> List[BaseMetric]:
        """
        Get all enabled metric instances.
        
        Returns:
            List of enabled BaseMetric instances.
        """
        enabled = []
        for name, config in self._configs.items():
            if config.enabled:
                enabled.append(self.get_metric(name))
        return enabled
    
    def get_enabled_names(self) -> List[str]:
        """
        Get names of all enabled metrics.
        
        Returns:
            List of enabled metric names.
        """
        return [name for name, config in self._configs.items() if config.enabled]
    
    def list_metrics(self, include_disabled: bool = False) -> List[str]:
        """
        List all registered metric names.
        
        Args:
            include_disabled: Whether to include disabled metrics.
        
        Returns:
            List of metric names.
        """
        if include_disabled:
            return list(self._configs.keys())
        return self.get_enabled_names()
    
    def enable_metric(self, name: str):
        """Enable a metric by name."""
        if name in self._configs:
            self._configs[name].enabled = True
    
    def disable_metric(self, name: str):
        """Disable a metric by name."""
        if name in self._configs:
            self._configs[name].enabled = False
    
    def set_enabled_metrics(self, names: List[str]):
        """
        Enable only the specified metrics, disable all others.
        
        Args:
            names: List of metric names to enable.
        """
        for name, config in self._configs.items():
            config.enabled = name in names
    
    def __len__(self) -> int:
        """Return number of registered metrics."""
        return len(self._configs)
    
    def __contains__(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._configs
    
    def __iter__(self):
        """Iterate over metric names."""
        return iter(self._configs.keys())
    
    def __repr__(self) -> str:
        enabled = len(self.get_enabled_names())
        return f"MetricRegistry(total={len(self)}, enabled={enabled})"


def create_default_registry(
    config_path: str = "config/metrics_config.yaml",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_concurrent: int = 10,
) -> MetricRegistry:
    """
    Create a MetricRegistry with default settings.
    
    Convenience function that sets up the client, cache, and registry.
    
    Args:
        config_path: Path to metrics YAML config.
        api_key: OpenAI API key (or set OPENAI_API_KEY env var).
        model: Model name to use.
        base_url: Optional custom API base URL.
        cache_dir: Optional cache directory (None = no caching).
        max_concurrent: Maximum concurrent API requests.
    
    Returns:
        Configured MetricRegistry.
    
    Example:
        >>> registry = create_default_registry(
        ...     config_path="config/metrics_config.yaml",
        ...     model="gpt-4o-mini",
        ...     cache_dir=".eval_cache",
        ... )
        >>> metrics = registry.get_enabled_metrics()
    """
    # Create API client
    client = OpenAIClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_concurrent=max_concurrent,
    )
    
    # Create cache if directory specified
    cache = None
    if cache_dir:
        cache = ResultCache(cache_dir=cache_dir, enabled=True)
    
    # Create and return registry
    return MetricRegistry.from_yaml(
        config_path=config_path,
        client=client,
        cache=cache,
    )
