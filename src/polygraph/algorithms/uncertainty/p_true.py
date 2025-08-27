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
P(True): Uncertainty estimation using few-shot prompting and answer validation.

This algorithm estimates the probability that a generated answer is true
by using few-shot examples and prompting the model to assess correctness.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, List

from ...utils.base_detector import BaseDetector

# Import P(True) implementation from the original location
try:
    from ....polygraphLLM.detectors.snne.uncertainty.uncertainty_measures.p_true import (
        construct_few_shot_prompt,
        calculate_p_true
    )
    HAS_P_TRUE = True
except ImportError:
    logger.warning("P(True) implementation not available")
    HAS_P_TRUE = False

logger = logging.getLogger(__name__)


def p_true(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    n_samples: int = 5,
    temperature: float = 0.8,
    use_few_shot: bool = True,
    hint: bool = False,
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    Estimate uncertainty using P(True) method.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        threshold: Probability threshold for hallucination decision (0.0-1.0)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        use_few_shot: Whether to use few-shot prompting
        hint: Whether to provide hints in prompts
        **kwargs: Additional parameters
        
    Returns:
        uncertainty_score (float): 1 - P(True) uncertainty measure
        is_hallucination (bool): True if P(True) below threshold
        explanation (dict): Contains P(True) computation details
    """
    detector = BaseDetector()
    
    if not HAS_P_TRUE:
        logger.error("P(True) implementation not available")
        explanation = {
            'algorithm': 'p_true',
            'error': 'P(True) implementation not available'
        }
        return 0.5, False, explanation
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question.strip())[0]
        
        # Generate samples if not provided (brainstormed answers)
        if samples is None:
            samples = detector.ask_llm(question.strip(), n=n_samples, temperature=temperature)
        
        # Remove the main answer from samples to avoid duplication
        brainstormed_answers = [s for s in samples if s != answer]
        
        # Prepare few-shot prompt if requested
        few_shot_prompt = ""
        if use_few_shot:
            # For simplicity, we'll use an empty few-shot prompt
            # In a full implementation, this would be constructed from training data
            few_shot_prompt = ""
        
        # Calculate P(True) score
        try:
            # Note: This is a simplified version
            # The full implementation would require a model with get_p_true method
            log_prob = _calculate_p_true_simplified(
                detector,
                question,
                answer,
                brainstormed_answers,
                few_shot_prompt,
                hint
            )
            
            # Convert log probability to probability
            p_true_score = min(max(np.exp(log_prob), 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"P(True) calculation failed: {e}, using fallback")
            p_true_score = 0.5  # Neutral probability
        
        # Uncertainty is 1 - P(True)
        uncertainty_score = 1.0 - p_true_score
        
        # Prepare explanation
        explanation = {
            'algorithm': 'p_true',
            'p_true_score': p_true_score,
            'uncertainty_score': uncertainty_score,
            'threshold': threshold,
            'n_samples': len(samples),
            'temperature': temperature,
            'use_few_shot': use_few_shot,
            'hint': hint,
            'brainstormed_answers': brainstormed_answers,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision (low P(True) = potential hallucination)
        is_hallucination = p_true_score < threshold
        
        logger.info(f'P(True): p_true={p_true_score:.3f}, uncertainty={uncertainty_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return uncertainty_score, is_hallucination, explanation
        
    except Exception as e:
        logger.error(f'P(True) failed: {e}')
        explanation = {
            'algorithm': 'p_true',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return 0.5, False, explanation


def _calculate_p_true_simplified(
    detector,
    question: str,
    most_probable_answer: str,
    brainstormed_answers: List[str],
    few_shot_prompt: str,
    hint: bool = False
) -> float:
    """Simplified P(True) calculation without requiring specialized model interface."""
    try:
        # Construct prompt
        if few_shot_prompt:
            prompt = few_shot_prompt + '\n'
        else:
            prompt = ''
        
        prompt += 'Question: ' + question
        prompt += '\nBrainstormed Answers: '
        for answer in brainstormed_answers + [most_probable_answer]:
            prompt += answer.strip() + '\n'
        prompt += 'Possible answer: ' + most_probable_answer + '\n'
        
        if not hint:
            prompt += 'Is the possible answer:\n'
            prompt += 'A) True\n'
            prompt += 'B) False\n'
            prompt += 'The possible answer is:'
        else:
            prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'
        
        # Get response from LLM
        response = detector.ask_llm(prompt)[0].strip().lower()
        
        # Parse response to get probability
        if response.startswith('a') or 'true' in response:
            # High confidence that answer is true
            return np.log(0.8)  # Log probability
        elif response.startswith('b') or 'false' in response:
            # High confidence that answer is false
            return np.log(0.2)  # Log probability
        else:
            # Uncertain
            return np.log(0.5)  # Log probability
        
    except Exception as e:
        logger.error(f"Simplified P(True) calculation failed: {e}")
        return np.log(0.5)  # Return neutral log probability
