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
Generation reader for loading _dump_generations output.

This module provides utilities to read JSONL files produced by the
_dump_generations method in ray_trainer.py.

Expected file structure:
    {data_dir}/
        {step}.jsonl       # e.g., 100.jsonl, 200.jsonl
    
    or nested:
    {data_dir}/
        rollout/
            {step}.jsonl
        monitor/
            {step}.jsonl
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from .schemas import Generation, GenerationDataset

logger = logging.getLogger(__name__)


class GenerationReader:
    """
    Read generations from _dump_generations output.
    
    Supports:
    - JSONL files: {step}.jsonl with one entry per line
    - Directory structure: rollout_data_dir/{rollout|monitor}/{step}.jsonl
    
    The reader is designed to be independent of the exact schema, allowing
    easy adaptation to changes in _dump_generations output format.
    
    Example:
        >>> reader = GenerationReader("/path/to/rollout_data_dir/rollout")
        >>> steps = reader.list_available_steps()
        >>> print(steps)  # [100, 200, 300, ...]
        >>> generations = reader.load_step(100)
        >>> print(len(generations))  # Number of samples at step 100
    """
    
    # Pattern to match step files: {step}.jsonl where step is an integer
    STEP_FILE_PATTERN = re.compile(r"^(\d+)\.jsonl$")
    
    def __init__(
        self,
        data_dir: str,
        required_fields: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Initialize the GenerationReader.
        
        Args:
            data_dir: Path to directory containing {step}.jsonl files.
            required_fields: List of required field names in each entry.
                Defaults to ["input", "output", "score", "step"].
            encoding: File encoding for reading JSONL files.
        """
        self.data_dir = Path(data_dir)
        self.required_fields = required_fields or ["input", "output", "score", "step"]
        self.encoding = encoding
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.data_dir}")
    
    def list_available_steps(self) -> List[int]:
        """
        List all available training steps.
        
        Returns:
            Sorted list of step numbers found in the data directory.
        """
        steps = []
        for filename in os.listdir(self.data_dir):
            match = self.STEP_FILE_PATTERN.match(filename)
            if match:
                steps.append(int(match.group(1)))
        return sorted(steps)
    
    def get_step_file_path(self, step: int) -> Path:
        """
        Get the file path for a specific step.
        
        Args:
            step: Training step number.
        
        Returns:
            Path to the JSONL file for this step.
        
        Raises:
            FileNotFoundError: If the step file doesn't exist.
        """
        filepath = self.data_dir / f"{step}.jsonl"
        if not filepath.exists():
            raise FileNotFoundError(f"No data file for step {step}: {filepath}")
        return filepath
    
    def load_step(self, step: int) -> List[Generation]:
        """
        Load all generations for a specific training step.
        
        Args:
            step: Training step number.
        
        Returns:
            List of Generation instances.
        
        Raises:
            FileNotFoundError: If the step file doesn't exist.
            json.JSONDecodeError: If JSONL parsing fails.
            KeyError: If required fields are missing.
        """
        filepath = self.get_step_file_path(step)
        generations = []
        
        with open(filepath, "r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    generation = Generation.from_dict(data, self.required_fields)
                    generations.append(generation)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at {filepath}:{line_num}: {e}")
                    raise
                except KeyError as e:
                    logger.warning(f"Missing field at {filepath}:{line_num}: {e}")
                    raise
        
        logger.debug(f"Loaded {len(generations)} generations from step {step}")
        return generations
    
    def load_steps(self, steps: List[int]) -> Dict[int, List[Generation]]:
        """
        Load generations for multiple steps.
        
        Args:
            steps: List of training step numbers.
        
        Returns:
            Dictionary mapping step -> list of generations.
        """
        result = {}
        for step in steps:
            try:
                result[step] = self.load_step(step)
            except FileNotFoundError:
                logger.warning(f"Step {step} not found, skipping")
        return result
    
    def load_all_steps(self) -> Dict[int, List[Generation]]:
        """
        Load all available steps.
        
        Returns:
            Dictionary mapping step -> list of generations.
        """
        steps = self.list_available_steps()
        return self.load_steps(steps)
    
    def load_as_dataset(
        self,
        steps: Optional[List[int]] = None,
        metadata: Optional[Dict] = None,
    ) -> GenerationDataset:
        """
        Load generations as a GenerationDataset.
        
        Args:
            steps: List of steps to load. If None, loads all available steps.
            metadata: Optional metadata to attach to the dataset.
        
        Returns:
            GenerationDataset containing all loaded generations.
        """
        if steps is None:
            step_data = self.load_all_steps()
        else:
            step_data = self.load_steps(steps)
        
        all_generations = []
        for step in sorted(step_data.keys()):
            all_generations.extend(step_data[step])
        
        return GenerationDataset(
            generations=all_generations,
            metadata=metadata or {"data_dir": str(self.data_dir)},
        )
    
    def count_generations(self, step: Optional[int] = None) -> Union[int, Dict[int, int]]:
        """
        Count generations without fully loading them.
        
        Args:
            step: If provided, count for specific step. Otherwise, count all.
        
        Returns:
            Either an int (for single step) or dict mapping step -> count.
        """
        if step is not None:
            filepath = self.get_step_file_path(step)
            with open(filepath, "r", encoding=self.encoding) as f:
                return sum(1 for line in f if line.strip())
        
        counts = {}
        for s in self.list_available_steps():
            filepath = self.data_dir / f"{s}.jsonl"
            with open(filepath, "r", encoding=self.encoding) as f:
                counts[s] = sum(1 for line in f if line.strip())
        return counts
    
    def __repr__(self) -> str:
        steps = self.list_available_steps()
        return f"GenerationReader(data_dir='{self.data_dir}', steps={len(steps)})"


def discover_generation_dirs(root_dir: str) -> Dict[str, GenerationReader]:
    """
    Discover all generation directories under a root directory.
    
    Useful for experiments that dump both rollout and monitor generations:
        {root_dir}/
            rollout/
                {step}.jsonl
            monitor/
                {step}.jsonl
    
    Args:
        root_dir: Root directory to search.
    
    Returns:
        Dictionary mapping subdirectory name -> GenerationReader.
    """
    root = Path(root_dir)
    readers = {}
    
    # Check if root itself contains step files
    if any(GenerationReader.STEP_FILE_PATTERN.match(f) for f in os.listdir(root)):
        readers["root"] = GenerationReader(root_dir)
    
    # Check subdirectories
    for subdir in root.iterdir():
        if subdir.is_dir():
            try:
                reader = GenerationReader(str(subdir))
                if reader.list_available_steps():
                    readers[subdir.name] = reader
            except (FileNotFoundError, NotADirectoryError):
                continue
    
    return readers
