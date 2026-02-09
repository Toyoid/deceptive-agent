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
Data schemas for retroactive evaluation.

These dataclasses define the structure for reading and processing
generation data from _dump_generations output.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Generation:
    """
    A single generation sample from _dump_generations output.
    
    Attributes:
        input: The input text (prompt) given to the model.
        output: The generated response from the model.
        score: The reward score assigned to this generation.
        step: The training step at which this generation was produced.
        system_info: System information/prompt provided to the model.
        extra: Additional fields from reward_extra_infos_dict or other sources.
        uid: Optional unique identifier for this generation (for caching).
    """
    input: str
    output: str
    score: float
    step: int
    system_info: str = ""
    
    extra: Dict[str, Any] = field(default_factory=dict)
    uid: Optional[str] = None
    
    def __post_init__(self):
        """Generate a UID if not provided, based on content hash."""
        if self.uid is None:
            import hashlib
            content = f"{self.step}:{self.input}:{self.output}"
            self.uid = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], required_fields: Optional[List[str]] = None) -> "Generation":
        """
        Create a Generation from a dictionary (e.g., from JSONL).
        
        Args:
            data: Dictionary containing generation data.
            required_fields: List of required field names. Defaults to ["input", "output", "score", "step"].
        
        Returns:
            Generation instance.
        
        Raises:
            KeyError: If a required field is missing.
        """
        if required_fields is None:
            required_fields = ["input", "output", "score", "step"]
        
        # Validate required fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise KeyError(f"Missing required fields: {missing}")
        
        # Extract core fields
        core_fields = {"input", "output", "score", "step", "system_info", "uid"}
        extra = {k: v for k, v in data.items() if k not in core_fields}
        
        return cls(
            input=data["input"],
            output=data["output"],
            score=float(data["score"]),
            step=int(data["step"]),
            system_info=data.get("system_info", ""),
            extra=extra,
            uid=data.get("uid"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "input": self.input,
            "output": self.output,
            "score": self.score,
            "step": self.step,
            "system_info": self.system_info,
            "uid": self.uid,
        }
        result.update(self.extra)
        return result


@dataclass
class GenerationDataset:
    """
    A collection of generations, typically from one or more training steps.
    
    Attributes:
        generations: List of Generation instances.
        metadata: Dataset-level metadata (e.g., model name, experiment config).
    """
    generations: List[Generation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.generations)
    
    def __iter__(self):
        return iter(self.generations)
    
    def __getitem__(self, idx):
        return self.generations[idx]
    
    def filter_by_step(self, step: int) -> "GenerationDataset":
        """Return a new dataset containing only generations from a specific step."""
        filtered = [g for g in self.generations if g.step == step]
        return GenerationDataset(generations=filtered, metadata=self.metadata)
    
    def get_steps(self) -> List[int]:
        """Get sorted list of unique steps in the dataset."""
        return sorted(set(g.step for g in self.generations))
    
    def get_scores(self) -> List[float]:
        """Get list of reward scores."""
        return [g.score for g in self.generations]
    
    def group_by_step(self) -> Dict[int, List[Generation]]:
        """Group generations by training step."""
        groups: Dict[int, List[Generation]] = {}
        for g in self.generations:
            if g.step not in groups:
                groups[g.step] = []
            groups[g.step].append(g)
        return groups
