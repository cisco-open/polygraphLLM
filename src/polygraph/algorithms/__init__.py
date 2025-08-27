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
Non-uncertainty based hallucination detection algorithms.

These algorithms analyze query-answer pairs and return explanations plus binary hallucination decisions.
"""

from .chainpoll import chainpoll
from .chatprotect import chatprotect  
from .geval import geval
from .refchecker import refchecker
from .selfcheckgpt import selfcheckgpt_bertscore, selfcheckgpt_ngram, selfcheckgpt_prompt, selfcheckgpt_mqag

__all__ = [
    'chainpoll',
    'chatprotect', 
    'geval',
    'refchecker',
    'selfcheckgpt_bertscore',
    'selfcheckgpt_ngram', 
    'selfcheckgpt_prompt',
    'selfcheckgpt_mqag'
]
