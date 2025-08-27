# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
PolyGraph: A comprehensive library for LLM hallucination detection.

Import structure:
    # Non-uncertainty algorithms
    from polygraph.algorithms import chainpoll, chatProtect, geval, refchecker, selfcheckgpt
    
    # Uncertainty-based algorithms
    from polygraph.algorithms.uncertainty import SNNE, TESNNE, kernel_uncertainty, p_true, semantic_entropy
    
    # Utilities
    from polygraph.utils import *
"""

__version__ = "1.0.0"
__author__ = "Cisco Systems, Inc."

# Legacy support - gradually deprecated
from .algorithms import *
from .utils import *
