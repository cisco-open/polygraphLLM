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

import logging
import os
import re

from .base import Detector

from ..prompts.default import (
    DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_COT_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT
)

logger = logging.getLogger(__name__)


class LLMUncertainty(Detector):
    id = 'llm_uncertainty'
    display_name = 'LLM-Uncertainty'

    def __init__(self):
        super().__init__()
        self.prompt_strategy = os.getenv("LLM_UNCERTAINTY_PROMPT_STRATEGY", "cot")


    def create_prompt(self, question, answer):
        mapping = {
            "cot": DEFAULT_LLM_UNCERTAINTY_COT_PROMPT,
            "vanilla": DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT,
            "self-probing": DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT,
            "multi-step": DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT
        }

        return mapping[self.prompt_strategy].format(question=question, answer=answer)

    def score(self, question, answer=None, samples=None, summary=None):
        if not answer:
            answer = self.ask_llm(question)[0]
        answer = answer.strip()
        prompt = self.create_prompt(question, answer)
        response = self.ask_llm(prompt, n=1)
        pattern = r'Overall Confidence: (\d+)%' if self.prompt_strategy == "multi-step" else r'Confidence: (\d+)%'

        match = re.search(pattern, response[0])
        confidence_level = 0
        if match:
            confidence_level = int(match.group(1))
        confidence_fraction = f"{confidence_level}/100"

        return confidence_fraction, answer, response
