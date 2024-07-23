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
        if match:
            confidence_level = int(match.group(1))
        confidence_fraction = f"{confidence_level}/100"

        return confidence_fraction, answer, response
