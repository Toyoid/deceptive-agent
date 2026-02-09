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
Custom OmegaConf resolvers for config interpolation.

This module registers custom resolvers that enable arithmetic and other
operations in YAML config files.

Usage in YAML:
    # Addition
    max_length: ${add:${data.max_prompt_length},${data.max_response_length}}
    
    # Subtraction
    remaining: ${sub:${total},${used}}
    
    # Multiplication
    total_tokens: ${mul:${batch_size},${seq_len}}
    
    # Integer division
    num_batches: ${div:${total_samples},${batch_size}}

Import this module before loading configs to register the resolvers:
    from verl.utils.config_resolvers import register_resolvers
    register_resolvers()
"""

from functools import reduce
from operator import mul as _mul

from omegaconf import OmegaConf

_RESOLVERS_REGISTERED = False


def _product(args):
    """Compute the product of all arguments."""
    return reduce(_mul, args, 1)


def register_resolvers():
    """
    Register custom OmegaConf resolvers for arithmetic operations.
    
    This function is idempotent - calling it multiple times is safe.
    """
    global _RESOLVERS_REGISTERED
    
    if _RESOLVERS_REGISTERED:
        return
    
    # Addition: ${add:a,b} or ${add:a,b,c,...}
    OmegaConf.register_new_resolver(
        "add",
        lambda *args: sum(args),
        replace=True
    )
    
    # Subtraction: ${sub:a,b} -> a - b
    OmegaConf.register_new_resolver(
        "sub",
        lambda a, b: a - b,
        replace=True
    )
    
    # Multiplication: ${mul:a,b} or ${mul:a,b,c,...}
    OmegaConf.register_new_resolver(
        "mul",
        lambda *args: _product(args),
        replace=True
    )
    
    # Integer division: ${div:a,b} -> a // b
    OmegaConf.register_new_resolver(
        "div",
        lambda a, b: a // b,
        replace=True
    )
    
    # Float division: ${fdiv:a,b} -> a / b
    OmegaConf.register_new_resolver(
        "fdiv",
        lambda a, b: a / b,
        replace=True
    )
    
    # Minimum: ${min:a,b} or ${min:a,b,c,...}
    OmegaConf.register_new_resolver(
        "min",
        lambda *args: min(args),
        replace=True
    )
    
    # Maximum: ${max:a,b} or ${max:a,b,c,...}
    OmegaConf.register_new_resolver(
        "max",
        lambda *args: max(args),
        replace=True
    )
    
    _RESOLVERS_REGISTERED = True
