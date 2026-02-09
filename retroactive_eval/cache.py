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
Result caching module for retroactive evaluation.

This module provides persistent caching of API responses to:
1. Avoid re-computation when re-running evaluations
2. Enable incremental evaluation (add new metrics without re-evaluating old ones)
3. Save API costs during development and debugging

Cache structure:
    {cache_dir}/
        {metric_name}/
            {step}/
                {generation_uid}.json
        metadata.json

Each cached result contains:
- generation_uid: Unique identifier for the generation
- metric_name: Name of the metric
- score: Computed score
- token_probs: Raw token probabilities from API
- timestamp: When the result was computed
- model: Model used for evaluation
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """A single cached evaluation result."""
    generation_uid: str
    metric_name: str
    score: float
    token_probs: Dict[str, float]
    step: int
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResult":
        """Create from dictionary."""
        return cls(**data)


class ResultCache:
    """
    Persistent cache for evaluation results.
    
    Caches are organized by metric name and step for efficient lookup.
    
    Example:
        >>> cache = ResultCache("/path/to/cache")
        >>> 
        >>> # Check if result exists
        >>> if cache.has_result("deception", generation_uid, step=100):
        ...     result = cache.get_result("deception", generation_uid, step=100)
        ... else:
        ...     # Compute result...
        ...     cache.save_result(result)
        >>> 
        >>> # Batch operations
        >>> existing_uids = cache.get_cached_uids("deception", step=100)
        >>> missing_uids = set(all_uids) - existing_uids
    """
    
    METADATA_FILE = "metadata.json"
    
    def __init__(
        self,
        cache_dir: str,
        enabled: bool = True,
        create_if_missing: bool = True,
    ):
        """
        Initialize the result cache.
        
        Args:
            cache_dir: Directory to store cached results.
            enabled: Whether caching is enabled. If False, all operations are no-ops.
            create_if_missing: Create cache directory if it doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if not enabled:
            logger.info("Result caching is disabled")
            return
        
        if create_if_missing:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
        # Load or create metadata
        self.metadata_path = self.cache_dir / self.METADATA_FILE
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized ResultCache at {cache_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {
            "created": time.time(),
            "version": "1.0",
            "metrics": {},
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_result_path(self, metric_name: str, step: int, generation_uid: str) -> Path:
        """Get the file path for a cached result."""
        return self.cache_dir / metric_name / str(step) / f"{generation_uid}.json"
    
    def _get_step_dir(self, metric_name: str, step: int) -> Path:
        """Get the directory for a metric/step combination."""
        return self.cache_dir / metric_name / str(step)
    
    def has_result(self, metric_name: str, generation_uid: str, step: int) -> bool:
        """Check if a result is cached."""
        if not self.enabled:
            return False
        return self._get_result_path(metric_name, step, generation_uid).exists()
    
    def get_result(
        self,
        metric_name: str,
        generation_uid: str,
        step: int,
    ) -> Optional[CachedResult]:
        """
        Retrieve a cached result.
        
        Args:
            metric_name: Name of the metric.
            generation_uid: Unique identifier for the generation.
            step: Training step.
        
        Returns:
            CachedResult if found, None otherwise.
        """
        if not self.enabled:
            return None
        
        path = self._get_result_path(metric_name, step, generation_uid)
        if not path.exists():
            return None
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return CachedResult.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cached result {path}: {e}")
            return None
    
    def save_result(self, result: CachedResult):
        """
        Save a result to cache.
        
        Args:
            result: CachedResult to save.
        """
        if not self.enabled:
            return
        
        path = self._get_result_path(result.metric_name, result.step, result.generation_uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Update metadata
        if result.metric_name not in self.metadata["metrics"]:
            self.metadata["metrics"][result.metric_name] = {
                "first_cached": time.time(),
                "steps": [],
            }
        if result.step not in self.metadata["metrics"][result.metric_name]["steps"]:
            self.metadata["metrics"][result.metric_name]["steps"].append(result.step)
            self.metadata["metrics"][result.metric_name]["steps"].sort()
            self._save_metadata()
    
    def save_results_batch(self, results: List[CachedResult]):
        """Save multiple results efficiently."""
        if not self.enabled:
            return
        
        for result in results:
            self.save_result(result)
    
    def get_cached_uids(self, metric_name: str, step: int) -> set:
        """
        Get all cached generation UIDs for a metric/step.
        
        Useful for identifying which generations need to be evaluated.
        
        Returns:
            Set of generation UIDs that have cached results.
        """
        if not self.enabled:
            return set()
        
        step_dir = self._get_step_dir(metric_name, step)
        if not step_dir.exists():
            return set()
        
        uids = set()
        for filename in os.listdir(step_dir):
            if filename.endswith(".json"):
                uids.add(filename[:-5])  # Remove .json extension
        return uids
    
    def get_all_results(
        self,
        metric_name: str,
        step: int,
    ) -> List[CachedResult]:
        """
        Get all cached results for a metric/step.
        
        Returns:
            List of CachedResult instances.
        """
        if not self.enabled:
            return []
        
        step_dir = self._get_step_dir(metric_name, step)
        if not step_dir.exists():
            return []
        
        results = []
        for filename in os.listdir(step_dir):
            if not filename.endswith(".json"):
                continue
            
            path = step_dir / filename
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                results.append(CachedResult.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        return results
    
    def get_cached_steps(self, metric_name: str) -> List[int]:
        """Get list of steps that have cached results for a metric."""
        if not self.enabled:
            return []
        
        metric_dir = self.cache_dir / metric_name
        if not metric_dir.exists():
            return []
        
        steps = []
        for dirname in os.listdir(metric_dir):
            try:
                steps.append(int(dirname))
            except ValueError:
                continue
        return sorted(steps)
    
    def get_cached_metrics(self) -> List[str]:
        """Get list of metrics that have cached results."""
        if not self.enabled:
            return []
        
        metrics = []
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                metrics.append(item.name)
        return sorted(metrics)
    
    def clear(self):
        """Clear all cached results."""
        if not self.enabled:
            return
        
        import shutil
        for metric_name in self.get_cached_metrics():
            metric_dir = self.cache_dir / metric_name
            if metric_dir.exists():
                shutil.rmtree(metric_dir)
        
        self.metadata["metrics"] = {}
        self._save_metadata()
        logger.info(f"Cleared all cache at {self.cache_dir}")
    
    def clear_metric(self, metric_name: str):
        """Clear all cached results for a metric."""
        if not self.enabled:
            return
        
        import shutil
        metric_dir = self.cache_dir / metric_name
        if metric_dir.exists():
            shutil.rmtree(metric_dir)
            logger.info(f"Cleared cache for metric: {metric_name}")
        
        if metric_name in self.metadata["metrics"]:
            del self.metadata["metrics"][metric_name]
            self._save_metadata()
    
    def clear_step(self, metric_name: str, step: int):
        """Clear cached results for a specific step."""
        if not self.enabled:
            return
        
        import shutil
        step_dir = self._get_step_dir(metric_name, step)
        if step_dir.exists():
            shutil.rmtree(step_dir)
            logger.info(f"Cleared cache for {metric_name}/step {step}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "metrics": {},
            "total_results": 0,
        }
        
        for metric_name in self.get_cached_metrics():
            metric_stats = {"steps": {}, "total": 0}
            for step in self.get_cached_steps(metric_name):
                count = len(self.get_cached_uids(metric_name, step))
                metric_stats["steps"][step] = count
                metric_stats["total"] += count
            stats["metrics"][metric_name] = metric_stats
            stats["total_results"] += metric_stats["total"]
        
        return stats
    
    def __repr__(self) -> str:
        if not self.enabled:
            return "ResultCache(enabled=False)"
        stats = self.get_stats()
        return (
            f"ResultCache(dir='{self.cache_dir}', "
            f"metrics={len(stats['metrics'])}, "
            f"total_results={stats['total_results']})"
        )


def generate_cache_key(
    input_text: str,
    output_text: str,
    step: int,
    metric_name: str,
) -> str:
    """
    Generate a unique cache key for a generation + metric combination.
    
    This is useful when generations don't have UIDs.
    """
    content = f"{step}:{metric_name}:{input_text}:{output_text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
