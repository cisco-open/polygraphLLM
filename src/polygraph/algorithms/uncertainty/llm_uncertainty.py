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
LLM Uncertainty: Uncertainty estimation using LLM self-assessment.

This algorithm asks the LLM to assess its own confidence in generated answers
using various prompting strategies.
"""

import logging
import re
from typing import Dict, Tuple, Optional, List

from ...utils.base_detector import BaseDetector
from ....polygraphLLM.prompts.default import (
    DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_COT_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT, 
    DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT
)

logger = logging.getLogger(__name__)


def llm_uncertainty(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    prompt_strategy: str = "cot",
    temperature: float = 0.7,
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    Estimate uncertainty using LLM self-assessment.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (unused)
        threshold: Confidence threshold for hallucination decision (0.0-1.0)
        prompt_strategy: Strategy for prompting ("vanilla", "cot", "self-probing", "multi-step")
        temperature: Temperature for confidence assessment
        **kwargs: Additional parameters
        
    Returns:
        uncertainty_score (float): Uncertainty measure (1 - confidence)
        is_hallucination (bool): True if confidence below threshold
        explanation (dict): Contains confidence analysis details
    """
    detector = BaseDetector()
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        answer = answer.strip()
        
        # Create confidence assessment prompt
        prompt = _create_prompt(question, answer, prompt_strategy)
        
        # Get confidence assessment
        response = detector.ask_llm(prompt, n=1, temperature=temperature)
        
        # Extract confidence level
        confidence_level = _extract_confidence(response[0], prompt_strategy)
        confidence_fraction = confidence_level / 100.0
        uncertainty_score = 1.0 - confidence_fraction
        
        # Prepare explanation
        explanation = {
            'algorithm': 'llm_uncertainty',
            'confidence_level': confidence_level,
            'confidence_fraction': confidence_fraction,
            'uncertainty_score': uncertainty_score,
            'threshold': threshold,
            'prompt_strategy': prompt_strategy,
            'temperature': temperature,
            'response': response[0],
            'answer': answer,
            'question': question
        }
        
        # Make binary decision (low confidence = potential hallucination)
        is_hallucination = confidence_fraction < threshold
        
        logger.info(f'LLM-Uncertainty: confidence={confidence_fraction:.3f}, uncertainty={uncertainty_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return uncertainty_score, is_hallucination, explanation
        
    except Exception as e:
        logger.error(f'LLM-Uncertainty failed: {e}')
        explanation = {
            'algorithm': 'llm_uncertainty',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return 0.5, False, explanation  # Return neutral uncertainty


def _create_prompt(question: str, answer: str, prompt_strategy: str) -> str:
    """Create confidence assessment prompt based on strategy."""
    mapping = {
        "cot": DEFAULT_LLM_UNCERTAINTY_COT_PROMPT,
        "vanilla": DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT,
        "self-probing": DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT,
        "multi-step": DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT
    }
    
    if prompt_strategy not in mapping:
        prompt_strategy = "cot"  # Default fallback
    
    return mapping[prompt_strategy].format(question=question, answer=answer)


def _extract_confidence(response: str, prompt_strategy: str) -> int:
    """Extract confidence percentage from response."""
    pattern = r'Overall Confidence: (\d+)%' if prompt_strategy == "multi-step" else r'Confidence: (\d+)%'
    
    match = re.search(pattern, response)
    if match:
        return int(match.group(1))
    
    # Fallback: try to find any percentage
    fallback_pattern = r'(\d+)%'
    matches = re.findall(fallback_pattern, response)
    if matches:
        return int(matches[-1])  # Take the last percentage found
    
    # Default to neutral confidence if no percentage found
    return 50


# Legacy class wrapper for backward compatibility
class LLMUncertainty(BaseDetector):
    """Legacy class wrapper for LLM Uncertainty algorithm."""
    
    id = 'llm_uncertainty'
    display_name = 'LLM-Uncertainty'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        prompt_strategy = "cot"
        temperature = 0.7
        
        if settings:
            prompt_strategy = settings.get("LLM_UNCERTAINTY_PROMPT_STRATEGY", "cot")
            temperature = float(settings.get("OPENAI_TEMPERATURE", 0.7))
        
        uncertainty_score, is_hallucination, explanation = llm_uncertainty(
            question=question,
            answer=answer,
            prompt_strategy=prompt_strategy,
            temperature=temperature
        )
        
        # Convert to legacy format (confidence fraction as string)
        confidence_level = explanation.get('confidence_level', 50)
        confidence_fraction = f"{confidence_level}/100"
        answer = explanation.get('answer', '')
        response = [explanation.get('response', '')]
        
        return confidence_fraction, answer, response
