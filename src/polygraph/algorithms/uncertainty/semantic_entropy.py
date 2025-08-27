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
Semantic Entropy: Uncertainty estimation using semantic clustering.

This algorithm computes uncertainty by clustering semantically equivalent responses
and measuring the entropy of the cluster assignment distribution.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, List

from ...utils.base_detector import BaseDetector

# Import semantic entropy implementation from the original location
try:
    from ....polygraphLLM.detectors.snne.uncertainty.uncertainty_measures.semantic_entropy import (
        get_semantic_ids_using_entailment,
        get_semantic_ids_using_embedding,
        cluster_assignment_entropy,
        predictive_entropy,
        EntailmentDeberta,
        SFR2Embedding
    )
    HAS_SEMANTIC_ENTROPY = True
except ImportError:
    logger.warning("Semantic entropy implementation not available")
    HAS_SEMANTIC_ENTROPY = False

logger = logging.getLogger(__name__)


def semantic_entropy(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    n_samples: int = 5,
    temperature: float = 0.8,
    clustering_method: str = "entailment",
    cluster_threshold: float = 0.5,
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    Estimate uncertainty using semantic entropy.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        threshold: Uncertainty threshold for hallucination decision (0.0-1.0)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        clustering_method: Method for semantic clustering ("entailment", "embedding")
        cluster_threshold: Threshold for clustering decisions
        **kwargs: Additional parameters
        
    Returns:
        uncertainty_score (float): Semantic entropy uncertainty measure
        is_hallucination (bool): True if uncertainty exceeds threshold
        explanation (dict): Contains entropy computation details
    """
    detector = BaseDetector()
    
    if not HAS_SEMANTIC_ENTROPY:
        logger.error("Semantic entropy implementation not available")
        explanation = {
            'algorithm': 'semantic_entropy',
            'error': 'Semantic entropy implementation not available'
        }
        return 0.5, False, explanation
    
    try:
        # Generate answer if not provided
        if not answer:
            answer = detector.ask_llm(question.strip())[0]
        
        # Generate samples if not provided
        if samples is None:
            samples = detector.ask_llm(question.strip(), n=n_samples, temperature=temperature)
        
        # Ensure we have the original answer in our samples
        if answer not in samples:
            samples = [answer] + samples
        
        # Ensure we have enough samples for meaningful computation
        if len(samples) < 2:
            logger.warning("Not enough samples for semantic entropy computation")
            explanation = {
                'algorithm': 'semantic_entropy',
                'error': 'Not enough samples',
                'answer': answer,
                'question': question
            }
            return 0.5, False, explanation
        
        # Compute semantic entropy
        entropy_score, cluster_info = _compute_semantic_entropy(
            samples,
            method=clustering_method,
            threshold=cluster_threshold
        )
        
        # Prepare explanation
        explanation = {
            'algorithm': 'semantic_entropy',
            'uncertainty_score': entropy_score,
            'threshold': threshold,
            'n_samples': len(samples),
            'temperature': temperature,
            'clustering_method': clustering_method,
            'cluster_threshold': cluster_threshold,
            'cluster_info': cluster_info,
            'samples': samples,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = entropy_score >= threshold
        
        logger.info(f'Semantic Entropy: uncertainty={entropy_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return entropy_score, is_hallucination, explanation
        
    except Exception as e:
        logger.error(f'Semantic entropy failed: {e}')
        explanation = {
            'algorithm': 'semantic_entropy',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return 0.5, False, explanation


def _compute_semantic_entropy(
    generations: List[str],
    method: str = "entailment",
    threshold: float = 0.5
) -> Tuple[float, Dict]:
    """Compute semantic entropy for a list of generations."""
    try:
        if method == "entailment":
            # Use entailment-based clustering
            model = EntailmentDeberta()
            semantic_ids = get_semantic_ids_using_entailment(
                generations, 
                model, 
                strict_entailment=False, 
                cluster_method='greedy'
            )
            cluster_info = {
                'method': 'entailment',
                'n_clusters': len(set(semantic_ids)),
                'semantic_ids': semantic_ids
            }
        elif method == "embedding":
            # Use embedding-based clustering
            model = SFR2Embedding()
            semantic_ids, similarity_matrix = get_semantic_ids_using_embedding(
                generations,
                model,
                cluster_method='dfs',
                threshold=threshold
            )
            cluster_info = {
                'method': 'embedding',
                'n_clusters': len(set(semantic_ids)),
                'semantic_ids': semantic_ids,
                'similarity_matrix': similarity_matrix.tolist() if hasattr(similarity_matrix, 'tolist') else None
            }
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Compute cluster assignment entropy
        entropy = cluster_assignment_entropy(semantic_ids)
        
        return entropy, cluster_info
        
    except Exception as e:
        logger.error(f"Semantic entropy computation failed: {e}")
        return 0.5, {'error': str(e)}  # Return neutral entropy on failure
