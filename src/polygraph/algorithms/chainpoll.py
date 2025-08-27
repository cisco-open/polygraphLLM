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
ChainPoll: Hallucination detection using multiple LLM responses and voting.

This algorithm generates multiple LLM responses to assess whether a given answer 
contains hallucinations by polling the model multiple times.
"""

import logging
import os
from typing import Dict, Tuple, Optional, List

from ..utils.base_detector import BaseDetector
from ...polygraphLLM.prompts.default import DEFAULT_CHAINPOLL_PROMPT

logger = logging.getLogger(__name__)


def chainpoll(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    n_samples: int = 5,
    temperature: float = 0.2,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using ChainPoll algorithm.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused in ChainPoll)
        samples: Pre-generated samples (unused in ChainPoll)
        n_samples: Number of samples for polling
        temperature: Temperature for LLM generation
        threshold: Threshold for hallucination decision (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains polling results and intermediate data
        is_hallucination (bool): True if hallucination detected
    """
    detector = BaseDetector()
    
    try:
        # Load prompt template
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '../../polygraphLLM/prompts/chainpoll.txt')
            with open(prompt_path, 'r') as pf:
                prompt_template = pf.read()
        except:
            prompt_template = DEFAULT_CHAINPOLL_PROMPT
        
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question.strip())[0]
        
        # Perform hallucination check via polling
        text = prompt_template.format(completion=answer, question=question)
        responses = detector.ask_llm(text, n_samples, temperature=temperature)
        logger.info(f'ChainPoll responses: {responses}')
        
        # Analyze responses
        hallucination_votes = [response.lower().startswith("yes") for response in responses]
        hallucination_score = hallucination_votes.count(True) / len(hallucination_votes)
        
        # Prepare explanation
        explanation = {
            'algorithm': 'chainpoll',
            'score': hallucination_score,
            'threshold': threshold,
            'n_samples': n_samples,
            'temperature': temperature,
            'responses': responses,
            'votes': hallucination_votes,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = hallucination_score >= threshold
        
        logger.info(f'ChainPoll: score={hallucination_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'ChainPoll failed: {e}')
        explanation = {
            'algorithm': 'chainpoll',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False  # Conservative default


# Legacy class wrapper for backward compatibility
class ChainPoll(BaseDetector):
    """Legacy class wrapper for ChainPoll algorithm."""
    
    id = 'chainpoll'
    display_name = 'Chain Poll'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        n_samples = 5
        if settings:
            n_samples = int(settings.get("CHAINPOLL_SAMPLING_NUMBER", 5))
        
        explanation, is_hallucination = chainpoll(
            question=question,
            answer=answer, 
            n_samples=n_samples
        )
        
        # Convert to legacy format
        score = explanation.get('score', 0.0)
        answer = explanation.get('answer', '')
        responses = explanation.get('responses', [])
        
        return score, answer, responses
