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
Unit tests for retroactive evaluation system.

Run with: pytest retroactive_eval/tests/ -v
"""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test data fixtures


@pytest.fixture
def sample_generation_data() -> List[Dict]:
    """Sample generation data for testing."""
    return [
        {
            "input": "User: What is 2+2?\nAssistant:",
            "output": " The answer is 4.",
            "score": 0.8,
            "step": 100,
        },
        {
            "input": "User: Tell me about the weather.\nAssistant:",
            "output": " It's sunny today.",
            "score": 0.6,
            "step": 100,
        },
        {
            "input": "User: What is the capital of France?\nAssistant:",
            "output": " Paris is the capital of France.",
            "score": 0.9,
            "step": 200,
        },
    ]


@pytest.fixture
def temp_data_dir(sample_generation_data) -> Path:
    """Create temporary directory with sample JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Group by step
        by_step = {}
        for item in sample_generation_data:
            step = item["step"]
            if step not in by_step:
                by_step[step] = []
            by_step[step].append(item)
        
        # Write JSONL files
        for step, items in by_step.items():
            filepath = tmpdir / f"{step}.jsonl"
            with open(filepath, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        
        yield tmpdir


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Data Schema Tests
# =============================================================================

def test_default_output_dir_is_sibling_of_rollout(tmp_path):
    """Output path construction must use native path semantics."""
    from retroactive_eval.run_eval import get_default_output_dir

    rollout_dir = tmp_path / "rollout_data" / "rollout"

    assert get_default_output_dir(rollout_dir) == tmp_path / "rollout_data" / "retro_eval"


class TestGeneration:
    """Tests for Generation dataclass."""
    
    def test_from_dict_basic(self):
        """Test creating Generation from dictionary."""
        from retroactive_eval.data_reader.schemas import Generation
        
        data = {
            "input": "Hello",
            "output": "World",
            "score": 0.5,
            "step": 100,
        }
        gen = Generation.from_dict(data)
        
        assert gen.input == "Hello"
        assert gen.output == "World"
        assert gen.score == 0.5
        assert gen.step == 100
        assert gen.uid is not None  # Auto-generated
    
    def test_from_dict_with_extra_fields(self):
        """Test that extra fields are captured."""
        from retroactive_eval.data_reader.schemas import Generation
        
        data = {
            "input": "Hello",
            "output": "World",
            "score": 0.5,
            "step": 100,
            "custom_field": "custom_value",
            "another_field": 123,
        }
        gen = Generation.from_dict(data)
        
        assert gen.extra["custom_field"] == "custom_value"
        assert gen.extra["another_field"] == 123
    
    def test_from_dict_missing_required(self):
        """Test error on missing required fields."""
        from retroactive_eval.data_reader.schemas import Generation
        
        data = {"input": "Hello", "output": "World"}  # Missing score and step
        
        with pytest.raises(KeyError):
            Generation.from_dict(data)
    
    def test_uid_generation_deterministic(self):
        """Test that UID is deterministic for same content."""
        from retroactive_eval.data_reader.schemas import Generation
        
        data = {
            "input": "Hello",
            "output": "World",
            "score": 0.5,
            "step": 100,
        }
        
        gen1 = Generation.from_dict(data)
        gen2 = Generation.from_dict(data)
        
        assert gen1.uid == gen2.uid
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        from retroactive_eval.data_reader.schemas import Generation
        
        gen = Generation(
            input="Hello",
            output="World",
            score=0.5,
            step=100,
            uid="test_uid",
        )
        
        d = gen.to_dict()
        assert d["input"] == "Hello"
        assert d["output"] == "World"
        assert d["score"] == 0.5
        assert d["step"] == 100
        assert d["uid"] == "test_uid"


class TestGenerationDataset:
    """Tests for GenerationDataset."""
    
    def test_filter_by_step(self, sample_generation_data):
        """Test filtering by step."""
        from retroactive_eval.data_reader.schemas import Generation, GenerationDataset
        
        generations = [Generation.from_dict(d) for d in sample_generation_data]
        dataset = GenerationDataset(generations=generations)
        
        filtered = dataset.filter_by_step(100)
        assert len(filtered) == 2
        assert all(g.step == 100 for g in filtered)
    
    def test_get_steps(self, sample_generation_data):
        """Test getting unique steps."""
        from retroactive_eval.data_reader.schemas import Generation, GenerationDataset
        
        generations = [Generation.from_dict(d) for d in sample_generation_data]
        dataset = GenerationDataset(generations=generations)
        
        steps = dataset.get_steps()
        assert steps == [100, 200]
    
    def test_group_by_step(self, sample_generation_data):
        """Test grouping by step."""
        from retroactive_eval.data_reader.schemas import Generation, GenerationDataset
        
        generations = [Generation.from_dict(d) for d in sample_generation_data]
        dataset = GenerationDataset(generations=generations)
        
        groups = dataset.group_by_step()
        assert len(groups[100]) == 2
        assert len(groups[200]) == 1


# =============================================================================
# Generation Reader Tests
# =============================================================================

class TestGenerationReader:
    """Tests for GenerationReader."""
    
    def test_list_available_steps(self, temp_data_dir):
        """Test listing available steps."""
        from retroactive_eval.data_reader.generation_reader import GenerationReader
        
        reader = GenerationReader(temp_data_dir)
        steps = reader.list_available_steps()
        
        assert 100 in steps
        assert 200 in steps
    
    def test_load_step(self, temp_data_dir):
        """Test loading a specific step."""
        from retroactive_eval.data_reader.generation_reader import GenerationReader
        
        reader = GenerationReader(temp_data_dir)
        generations = reader.load_step(100)
        
        assert len(generations) == 2
        assert all(g.step == 100 for g in generations)
    
    def test_load_nonexistent_step(self, temp_data_dir):
        """Test loading nonexistent step raises error."""
        from retroactive_eval.data_reader.generation_reader import GenerationReader
        
        reader = GenerationReader(temp_data_dir)
        
        with pytest.raises(FileNotFoundError):
            reader.load_step(999)
    
    def test_load_all_steps(self, temp_data_dir):
        """Test loading all steps."""
        from retroactive_eval.data_reader.generation_reader import GenerationReader
        
        reader = GenerationReader(temp_data_dir)
        step_to_generations = reader.load_all_steps()
        
        # Returns dict mapping step -> list of generations
        assert 100 in step_to_generations
        assert 200 in step_to_generations
        total_gens = sum(len(gens) for gens in step_to_generations.values())
        assert total_gens == 3


# =============================================================================
# Scoring Tests (now in OpenAIClient)
# =============================================================================

class TestScoring:
    """Tests for score computation in OpenAIClient."""
    
    def test_compute_score_basic(self):
        """Test basic score computation."""
        from retroactive_eval.clients.openai_client import OpenAIClient
        
        # Test the static scoring logic
        token_probs = {"0": 0.1, "1": 0.2, "2": 0.7}
        token_weights = [0.0, 0.5, 1.0]
        
        # Score = 0.0*0.1 + 0.5*0.2 + 1.0*0.7 = 0.8
        score = sum(p * w for p, w in zip(token_probs.values(), token_weights))
        assert abs(score - 0.8) < 0.001
    
    def test_six_token_scale(self):
        """Test with 0-5 scale (6 tokens)."""
        # Default scale
        valid_tokens = ["0", "1", "2", "3", "4", "5"]
        token_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # All probability on "5"
        token_probs = {t: 0.0 for t in valid_tokens}
        token_probs["5"] = 1.0
        
        score = sum(p * w for p, w in zip(token_probs.values(), token_weights))
        assert abs(score - 1.0) < 0.001
        
        # All probability on "2" (score = 0.4)
        token_probs = {t: 0.0 for t in valid_tokens}
        token_probs["2"] = 1.0
        
        score = sum(p * w for p, w in zip(token_probs.values(), token_weights))
        assert abs(score - 0.4) < 0.001


# =============================================================================
# Cache Tests
# =============================================================================

class TestResultCache:
    """Tests for ResultCache."""
    
    def test_save_and_load(self, temp_cache_dir):
        """Test saving and loading cached results."""
        from retroactive_eval.cache import ResultCache, CachedResult
        
        cache = ResultCache(temp_cache_dir)
        
        result = CachedResult(
            generation_uid="test_uid_123",
            metric_name="deception",
            score=0.75,
            token_probs={"0": 0.1, "5": 0.3, "10": 0.6},
            step=100,
            model="gpt-4o",
        )
        
        cache.save_result(result)
        
        # Load it back
        loaded = cache.get_result("deception", "test_uid_123", 100)
        
        assert loaded is not None
        assert loaded.generation_uid == "test_uid_123"
        assert loaded.metric_name == "deception"
        assert loaded.score == 0.75
        assert loaded.step == 100
    
    def test_load_nonexistent(self, temp_cache_dir):
        """Test loading nonexistent cache returns None."""
        from retroactive_eval.cache import ResultCache
        
        cache = ResultCache(temp_cache_dir)
        
        result = cache.get_result("nonexistent", "fake_uid", 999)
        assert result is None
    
    def test_exists(self, temp_cache_dir):
        """Test checking if cache exists."""
        from retroactive_eval.cache import ResultCache, CachedResult
        
        cache = ResultCache(temp_cache_dir)
        
        assert not cache.has_result("deception", "test_uid", 100)
        
        result = CachedResult(
            generation_uid="test_uid",
            metric_name="deception",
            score=0.5,
            token_probs={},
            step=100,
        )
        cache.save_result(result)
        
        assert cache.has_result("deception", "test_uid", 100)
    
    def test_clear_metric(self, temp_cache_dir):
        """Test clearing cache for a specific metric."""
        from retroactive_eval.cache import ResultCache, CachedResult
        
        cache = ResultCache(temp_cache_dir)
        
        # Save some results
        for i in range(3):
            result = CachedResult(
                generation_uid=f"uid_{i}",
                metric_name="deception",
                score=0.5,
                token_probs={},
                step=100,
            )
            cache.save_result(result)
        
        assert cache.has_result("deception", "uid_0", 100)
        
        cache.clear_metric("deception")
        
        assert not cache.has_result("deception", "uid_0", 100)


# =============================================================================
# Aggregator Tests
# =============================================================================

class TestMetricsAggregator:
    """Tests for MetricsAggregator."""
    
    def test_add_and_get_statistics(self):
        """Test adding results and computing statistics."""
        from retroactive_eval.analysis.aggregator import MetricsAggregator
        from retroactive_eval.metrics.base_metric import MetricResult
        from retroactive_eval.data_reader.schemas import Generation
        
        aggregator = MetricsAggregator()
        
        # Add some results
        for i in range(10):
            result = MetricResult(
                generation_uid=f"uid_{i}",
                metric_name="deception",
                score=i / 10,  # 0.0 to 0.9
                token_probs={},
                step=100,
            )
            gen = Generation(
                input="test",
                output="test",
                score=0.5,
                step=100,
                uid=f"uid_{i}",
            )
            aggregator.add_result(result, gen)
        
        stats = aggregator.get_statistics("deception", 100)
        
        assert stats.count == 10
        assert abs(stats.mean - 0.45) < 0.01  # Mean of 0.0-0.9
        assert stats.min_val == 0.0
        assert stats.max_val == 0.9
    
    def test_get_time_series(self):
        """Test getting time series data."""
        from retroactive_eval.analysis.aggregator import MetricsAggregator
        from retroactive_eval.metrics.base_metric import MetricResult
        from retroactive_eval.data_reader.schemas import Generation
        
        aggregator = MetricsAggregator()
        
        # Add results for multiple steps
        for step in [100, 200, 300]:
            for i in range(5):
                result = MetricResult(
                    generation_uid=f"uid_{step}_{i}",
                    metric_name="deception",
                    score=step / 1000 + i / 10,
                    token_probs={},
                    step=step,
                )
                gen = Generation(
                    input="test",
                    output="test",
                    score=0.5,
                    step=step,
                    uid=f"uid_{step}_{i}",
                )
                aggregator.add_result(result, gen)
        
        ts = aggregator.get_time_series("deception")
        
        assert ts.steps == [100, 200, 300]
        assert len(ts.means) == 3
        assert len(ts.stds) == 3
    
    def test_multiple_metrics(self):
        """Test handling multiple metrics."""
        from retroactive_eval.analysis.aggregator import MetricsAggregator
        from retroactive_eval.metrics.base_metric import MetricResult
        from retroactive_eval.data_reader.schemas import Generation
        
        aggregator = MetricsAggregator()
        
        # Add results for multiple metrics
        for metric in ["deception", "omission"]:
            result = MetricResult(
                generation_uid="uid_1",
                metric_name=metric,
                score=0.5 if metric == "deception" else 0.3,
                token_probs={},
                step=100,
            )
            gen = Generation(
                input="test",
                output="test",
                score=0.5,
                step=100,
                uid="uid_1",
            )
            aggregator.add_result(result, gen)
        
        metrics = aggregator.get_metrics()
        assert "deception" in metrics
        assert "omission" in metrics


# =============================================================================
# Metric Config Tests
# =============================================================================

class TestMetricConfig:
    """Tests for MetricConfig."""
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        from retroactive_eval.metrics.base_metric import MetricConfig
        
        data = {
            "system_prompt": "You are a judge.",
            "prompt_suffix": "Rate from 0 to 10.",
            "valid_tokens": ["0", "1", "2"],
            "token_weights": [0.0, 0.5, 1.0],
        }
        
        config = MetricConfig.from_dict("test_metric", data)
        
        assert config.name == "test_metric"
        assert config.system_prompt == "You are a judge."
        assert config.valid_tokens == ["0", "1", "2"]
    
    def test_from_dict_with_defaults(self):
        """Test creating config with defaults."""
        from retroactive_eval.metrics.base_metric import MetricConfig
        
        data = {
            "system_prompt": "You are a judge.",
            "prompt_suffix": "Rate.",
        }
        defaults = {
            "valid_tokens": ["0", "5", "10"],
            "token_weights": [0.0, 0.5, 1.0],
        }
        
        config = MetricConfig.from_dict("test_metric", data, defaults)
        
        assert config.valid_tokens == ["0", "5", "10"]
        assert config.token_weights == [0.0, 0.5, 1.0]


# =============================================================================
# Integration Tests (with mocked API)
# =============================================================================

class TestIntegration:
    """Integration tests with mocked API calls."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self, temp_data_dir, temp_cache_dir):
        """Test full evaluation pipeline with mocked API."""
        from retroactive_eval.data_reader.generation_reader import GenerationReader
        from retroactive_eval.metrics.base_metric import BaseMetric, MetricConfig
        from retroactive_eval.analysis.aggregator import MetricsAggregator
        from retroactive_eval.cache import ResultCache
        
        # Mock the OpenAI client
        mock_client = MagicMock()
        # Return (score, token_probs) tuple
        mock_client.get_score = AsyncMock(return_value=(
            0.5,  # score
            {"0": 0.1, "1": 0.1, "2": 0.3, "3": 0.3, "4": 0.1, "5": 0.1}  # token_probs
        ))
        mock_client.model = "test-model"
        
        # Setup
        reader = GenerationReader(temp_data_dir)
        cache = ResultCache(temp_cache_dir)
        aggregator = MetricsAggregator()
        
        config = MetricConfig(
            name="test_metric",
            system_prompt="You are a test judge.",
            prompt_suffix="Rate from 0 to 5.",
            valid_tokens=["0", "1", "2", "3", "4", "5"],
            token_weights=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        
        metric = BaseMetric(config, mock_client, cache)
        
        # Run evaluation - load_step returns List[Generation]
        generations = reader.load_step(100)
        
        for gen in generations:
            result = await metric.evaluate(gen)
            aggregator.add_result(result, gen)
        
        # Check results
        stats = aggregator.get_statistics("test_metric", 100)
        assert stats.count == 2
        assert not math.isnan(stats.mean)
        
        # Check caching worked - use correct method: has_result(metric, uid, step)
        assert cache.has_result("test_metric", generations[0].uid, 100)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
