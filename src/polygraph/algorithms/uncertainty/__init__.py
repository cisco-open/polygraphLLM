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
Uncertainty-based hallucination detection algorithms.

These algorithms compute uncertainty measures that can be thresholded for hallucination detection.
"""

from .snne import SNNE, TESNNE
from .llm_uncertainty import llm_uncertainty
from .semantic_entropy import semantic_entropy
from .kernel_uncertainty import kernel_uncertainty  
from .p_true import p_true

__all__ = [
    'SNNE',
    'TESNNE', 
    'llm_uncertainty',
    'semantic_entropy',
    'kernel_uncertainty',
    'p_true'
]
