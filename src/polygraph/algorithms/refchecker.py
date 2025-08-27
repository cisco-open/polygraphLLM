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
RefChecker: Reference-based hallucination detection.

This algorithm checks claims in generated text against retrieved reference documents
using triplet extraction and entailment analysis.
"""

import logging
from typing import Dict, Tuple, Optional, List

from ..utils.base_detector import BaseDetector

logger = logging.getLogger(__name__)


def refchecker(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[Dict, bool]:
    """
    Detect hallucinations using reference checking.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused in RefChecker)
        samples: Pre-generated samples (unused in RefChecker)
        threshold: Threshold for contradiction ratio (0.0-1.0)
        **kwargs: Additional parameters
        
    Returns:
        explanation (dict): Contains entailment analysis results
        is_hallucination (bool): True if contradiction ratio exceeds threshold
    """
    detector = BaseDetector()
    
    try:
        # Generate question if not provided
        if not question:
            question = detector.generate_question(answer)
        
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question.strip())[0]
        
        # Extract triplets from the answer
        triplets = detector.extract_triplets(answer, question, max_new_tokens=200)
        
        # Retrieve reference documents
        reference = detector.retrieve([question])
        
        # Check each triplet against references
        results = [
            detector.check(triplet, reference, question=question)
            for triplet in triplets
        ]
        
        # Aggregate results
        agg_results = _soft_agg(results)
        
        # Format scores
        for k, v in agg_results.items():
            agg_results[k] = float("{:.2f}".format(v))
        
        # Calculate contradiction ratio for hallucination detection
        contradiction_ratio = agg_results.get('Contradiction', 0.0)
        
        # Prepare explanation
        explanation = {
            'algorithm': 'refchecker',
            'entailment_results': agg_results,
            'contradiction_ratio': contradiction_ratio,
            'threshold': threshold,
            'triplets': triplets,
            'reference_docs': reference,
            'individual_results': results,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = contradiction_ratio >= threshold
        
        logger.info(f'RefChecker: contradiction_ratio={contradiction_ratio:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return explanation, is_hallucination
        
    except Exception as e:
        logger.error(f'RefChecker failed: {e}')
        explanation = {
            'algorithm': 'refchecker',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return explanation, False


def _soft_agg(results: List[str]) -> Dict[str, float]:
    """Aggregate results by taking the ratio of each category."""
    if not results:
        return {
            "Entailment": 0.0,
            "Neutral": 0.0,
            "Contradiction": 0.0,
            "Abstain": 1.0,
        }
    
    total = len(results)
    agg = {
        "Entailment": 0.0,
        "Neutral": 0.0,
        "Contradiction": 0.0,
        "Abstain": 0.0,
    }
    
    for result in results:
        if result in agg:
            agg[result] += 1.0
        else:
            agg["Abstain"] += 1.0  # Default unknown results to abstain
    
    for key in agg:
        agg[key] /= total
    
    return agg


# Legacy class wrapper for backward compatibility
class RefChecker(BaseDetector):
    """Legacy class wrapper for RefChecker algorithm."""
    
    id = 'refchecker'
    display_name = 'RefChecker'
    
    def score(self, question=None, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        explanation, is_hallucination = refchecker(
            question=question,
            answer=answer
        )
        
        # Convert to legacy format
        agg_results = explanation.get('entailment_results', {})
        answer = explanation.get('answer', '')
        
        return agg_results, answer, None
