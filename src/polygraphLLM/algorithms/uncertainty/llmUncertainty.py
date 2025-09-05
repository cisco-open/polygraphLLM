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
import re

from ..base import Detector

from polygraphLLM.utils.prompts.default import (
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


    def create_prompt(self, question, answer, prompt_strategy):
        mapping = {
            "cot": DEFAULT_LLM_UNCERTAINTY_COT_PROMPT,
            "vanilla": DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT,
            "self-probing": DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT,
            "multi-step": DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT
        }

        return mapping[prompt_strategy].format(question=question, answer=answer)

    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        if not answer:
            answer = self.ask_llm(question)[0]
        answer = answer.strip()
        prompt_strategy = self.find_settings_value(settings, "LLM_UNCERTAINTY_PROMPT_STRATEGY")
        temperature = float(self.find_settings_value(settings, "OPENAI_TEMPERATURE"))
        
        prompt = self.create_prompt(question, answer, prompt_strategy)
        response = self.ask_llm(prompt, n=1, temperature=temperature)
        pattern = r'Overall Confidence: (\d+)%' if prompt_strategy == "multi-step" else r'Confidence: (\d+)%'

        match = re.search(pattern, response[0])
        confidence_level = 0
        if match:
            confidence_level = int(match.group(1))
        confidence_fraction = f"{confidence_level}/100"

        return confidence_fraction, answer, response
    
    def detect_hallucination(self, question, answer=None, samples=None, summary=None, settings=None, threshold=0.5):
        """
        Detect hallucination based on threshold. Lower confidence indicates hallucination.
        
        Returns:
            tuple: (is_hallucinated: bool, raw_score: float, answer: str, additional_data)
        """
        confidence_fraction, answer, response = self.score(question, answer, samples, summary, settings)
        # Parse confidence as a number (e.g., "75/100" -> 0.75)
        try:
            confidence_score = float(confidence_fraction.split('/')[0]) / 100.0
        except:
            confidence_score = 0.5  # Default if parsing fails
        
        # Lower confidence indicates higher chance of hallucination
        is_hallucinated = bool(confidence_score < threshold)
        return is_hallucinated, confidence_score, answer, response