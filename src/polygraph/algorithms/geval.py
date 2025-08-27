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
G-Eval: Multi-dimensional quality evaluation for text generation.

This algorithm evaluates generated text across multiple dimensions:
coherence, consistency, fluency, and relevance.
"""

import logging
import os
import re
from typing import Dict, Tuple, Optional, List

from ..utils.base_detector import BaseDetector
from ...polygraphLLM.prompts.default import DEFAULT_COH_PROMPT, DEFAULT_FLU_PROMPT, DEFAUL_REL_PROMPT, DEFAULT_CON_PROMPT

logger = logging.getLogger(__name__)


def geval(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    summary: str = None,
    n_samples: int = 3,
    threshold: float = 0.5,
    metrics: List[str] = None,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Evaluate text quality using G-Eval across multiple dimensions.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused in G-Eval)
        samples: Pre-generated samples (unused in G-Eval)
        summary: Summary for evaluation (if None, generates one)
        n_samples: Number of samples for evaluation
        threshold: Threshold for overall quality (0.0-1.0)
        metrics: List of metrics to evaluate ['coherence', 'consistency', 'fluency', 'relevance']
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains quality scores across dimensions
        is_hallucination (bool): True if overall quality below threshold
    """
    detector = BaseDetector()
    
    if metrics is None:
        metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question)[0]
        answer = answer.strip()
        
        # Generate summary if not provided
        if not summary:
            summary_prompt = f"Create a summary with 20 maximum words from {answer}"
            summary = detector.ask_llm(summary_prompt)[0].strip()
        
        # Evaluate each metric
        scores = {}
        all_samples = []
        
        for metric in metrics:
            prompt = _create_prompt(answer, summary, metric)
            responses = detector.ask_llm(prompt, n_samples)
            all_samples.append(responses)
            
            # Parse scores from responses
            parsed_scores = [_parse_output(response.strip()) for response in responses]
            avg_score = sum(parsed_scores) / len(parsed_scores)
            normalized_score = _normalize_score(avg_score)
            
            scores[metric.title()] = float("{:.2f}".format(normalized_score))
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        scores['Total'] = float("{:.2f}".format(overall_score))
        
        # Prepare explanation
        explanation = {
            'algorithm': 'geval',
            'scores': scores,
            'metrics': metrics,
            'threshold': threshold,
            'n_samples': n_samples,
            'answer': answer,
            'summary': summary,
            'question': question,
            'samples': all_samples
        }
        
        # Make binary decision (low quality = potential hallucination)
        is_hallucination = overall_score < threshold
        
        logger.info(f'G-Eval: overall_score={overall_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'G-Eval failed: {e}')
        explanation = {
            'algorithm': 'geval',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def _create_prompt(answer: str, summary: str, metric: str) -> str:
    """Create evaluation prompt for a specific metric."""
    mapping = {
        "coherence": ("coh_detailed.txt", DEFAULT_COH_PROMPT),
        "consistency": ("con_detailed.txt", DEFAULT_CON_PROMPT),
        "fluency": ("flu_detailed.txt", DEFAULT_FLU_PROMPT),
        "relevance": ("rel_detailed.txt", DEFAUL_REL_PROMPT)
    }
    
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), f'../../polygraphLLM/prompts/{mapping[metric][0]}')
        with open(prompt_path) as readfile:
            prompt = readfile.read()
    except:
        prompt = mapping[metric][1]
    
    cur_prompt = prompt.replace('{{Document}}', answer).replace('{{Summary}}', summary)
    return cur_prompt


def _parse_output(output: str) -> float:
    """Parse numerical score from model output."""
    if ':' in output:
        output = output.rsplit(':', 1)[-1]
        
    matched = re.search("^ ?([\d\.]+)", output)
    if matched:
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        if ':' in output:
            output = output.rsplit(':', 1)[-1]
            matched = re.search("^ ?([\d\.]+)", output)
            if matched:
                try:
                    score = float(matched.group(1))
                except:
                    score = 0
        else:
            score = 0
    return score


def _normalize_score(score: float) -> float:
    """Normalize score to 0-1 range."""
    max_score = 5  # Maximum possible score
    normalized_score = score / max_score
    return normalized_score


# Legacy class wrapper for backward compatibility
class GEval(BaseDetector):
    """Legacy class wrapper for G-Eval algorithm."""
    
    id = 'g_eval'
    display_name = 'G-Eval'
    metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        n_samples = 3
        if settings:
            n_samples = int(settings.get("GEVAL_SAMPLING_NUMBER", 3))
        
        explanation, is_hallucination = geval(
            question=question,
            answer=answer,
            summary=summary,
            n_samples=n_samples,
            metrics=self.metrics
        )
        
        # Convert to legacy format
        scores = explanation.get('scores', {})
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return scores, answer, samples
