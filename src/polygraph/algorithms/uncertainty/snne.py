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
SNNE: Soft Nearest Neighbor Entropy for uncertainty estimation.

This algorithm computes uncertainty by measuring the entropy in the distribution
of semantic similarities between generated responses.
"""

import logging
import torch
import evaluate
from rouge_score import tokenizers
from sentence_transformers import SentenceTransformer
from typing import Dict, Tuple, Optional, List

from ...utils.base_detector import BaseDetector

# Import SNNE implementation from the original location
try:
    from ....polygraphLLM.detectors.snne.uncertainty.uncertainty_measures.semantic_entropy import (
        EntailmentDeberta, soft_nearest_neighbor_loss
    )
    HAS_SNNE = True
except ImportError:
    logger.warning("SNNE implementation not available")
    HAS_SNNE = False

logger = logging.getLogger(__name__)


def SNNE(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    n_samples: int = 5,
    temperature: float = 0.8,
    variant: str = 'only_denom',
    snne_temperature: float = 1.0,
    selfsim: bool = True,
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    Estimate uncertainty using Soft Nearest Neighbor Entropy (SNNE).
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        threshold: Uncertainty threshold for hallucination decision (0.0-1.0)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        variant: SNNE variant ('only_denom', etc.)
        snne_temperature: Temperature for SNNE computation
        selfsim: Whether to include self-similarity
        **kwargs: Additional parameters
        
    Returns:
        uncertainty_score (float): SNNE uncertainty measure
        is_hallucination (bool): True if uncertainty exceeds threshold
        explanation (dict): Contains SNNE computation details
    """
    detector = BaseDetector()
    
    if not HAS_SNNE:
        logger.error("SNNE implementation not available")
        explanation = {
            'algorithm': 'snne',
            'error': 'SNNE implementation not available'
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
            logger.warning("Not enough samples for SNNE computation")
            explanation = {
                'algorithm': 'snne',
                'error': 'Not enough samples',
                'answer': answer,
                'question': question
            }
            return 0.5, False, explanation
        
        # Compute SNNE score
        snne_score = _compute_snne_score(
            samples, 
            variant=variant,
            temperature=snne_temperature,
            selfsim=selfsim
        )
        
        # Prepare explanation
        explanation = {
            'algorithm': 'snne',
            'uncertainty_score': snne_score,
            'threshold': threshold,
            'n_samples': len(samples),
            'generation_temperature': temperature,
            'snne_temperature': snne_temperature,
            'variant': variant,
            'selfsim': selfsim,
            'samples': samples,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = snne_score >= threshold
        
        logger.info(f'SNNE: uncertainty={snne_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return snne_score, is_hallucination, explanation
        
    except Exception as e:
        logger.error(f'SNNE failed: {e}')
        explanation = {
            'algorithm': 'snne',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return 0.5, False, explanation


def TESNNE(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    TESNNE: Temperature-scaled SNNE variant.
    
    This is a simplified implementation that adjusts the SNNE computation
    with different temperature scaling.
    """
    # For now, TESNNE is the same as SNNE with different default parameters
    return SNNE(
        question=question,
        answer=answer,
        context=context,
        samples=samples,
        threshold=threshold,
        snne_temperature=10.0,  # Higher temperature for TESNNE
        **kwargs
    )


def _compute_snne_score(
    generations: List[str], 
    semantic_ids: List[int] = None, 
    variant: str = 'only_denom',
    temperature: float = 1.0, 
    selfsim: bool = True
) -> float:
    """Compute SNNE score for a list of generations."""
    try:
        # Initialize models lazily
        entailment_model = _get_entailment_model()
        embedding_model = _get_embedding_model()
        
        if entailment_model is None or embedding_model is None:
            logger.error("Required models not available for SNNE computation")
            return 0.5  # Return neutral score
        
        if semantic_ids is None:
            semantic_ids = list(range(len(generations)))
        
        # Compute lexical similarity matrix
        similarity_matrix = _compute_lexical_similarity(generations)
        
        # Compute SNNE
        snne = soft_nearest_neighbor_loss(
            generations,
            entailment_model, 
            embedding_model, 
            semantic_ids,
            similarity_matrix=similarity_matrix,
            variant=variant, 
            temperature=temperature, 
            exclude_diagonal=not selfsim
        ).item()
        
        return snne
        
    except Exception as e:
        logger.error(f"SNNE computation failed: {e}")
        return 0.5  # Return neutral score on failure


def _get_entailment_model():
    """Get entailment model with lazy initialization."""
    try:
        return EntailmentDeberta()
    except Exception as e:
        logger.error(f"Failed to initialize EntailmentDeberta: {e}")
        return None


def _get_embedding_model():
    """Get embedding model with lazy initialization."""
    try:
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # ~22MB, fastest
            "sentence-transformers/paraphrase-MiniLM-L6-v2",  # ~22MB
            "sentence-transformers/all-MiniLM-L12-v2",  # ~33MB
            "sentence-transformers/all-mpnet-base-v2",  # ~420MB
        ]
        
        for model_name in embedding_models:
            try:
                embedding_model = SentenceTransformer(
                    model_name,
                    device='mps' if torch.backends.mps.is_available() else 'cpu'
                )
                if hasattr(embedding_model, 'max_seq_length'):
                    embedding_model.max_seq_length = min(512, getattr(embedding_model, 'max_seq_length', 512))
                
                logger.info(f"Successfully loaded embedding model: {model_name}")
                return embedding_model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
                
        logger.error("Failed to load any embedding model")
        return None
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return None


def _compute_lexical_similarity(generations: List[str]) -> torch.Tensor:
    """Compute lexical similarity matrix for generations."""
    try:
        rouge = evaluate.load('rouge', keep_in_memory=True)
        
        similarity_matrix = []
        for i, gen_i in enumerate(generations):
            row = []
            for j, gen_j in enumerate(generations):
                if i == j:
                    row.append(1.0)
                else:
                    try:
                        # Compute ROUGE-L similarity as a proxy for lexical similarity
                        rouge_scores = rouge.compute(
                            predictions=[gen_i],
                            references=[gen_j],
                            use_stemmer=False
                        )
                        row.append(rouge_scores['rougeL'])
                    except Exception:
                        # Fallback to simple Jaccard similarity
                        words_i = set(gen_i.lower().split())
                        words_j = set(gen_j.lower().split())
                        intersection = len(words_i.intersection(words_j))
                        union = len(words_i.union(words_j))
                        similarity = intersection / union if union > 0 else 0.0
                        row.append(similarity)
            similarity_matrix.append(row)
        return torch.tensor(similarity_matrix)
        
    except Exception as e:
        logger.warning(f"Lexical similarity computation failed: {e}")
        # Fallback to identity matrix
        n = len(generations)
        return torch.eye(n)


# Legacy class wrapper for backward compatibility
class SNNEDetector(BaseDetector):
    """Legacy class wrapper for SNNE algorithm."""
    
    id = 'snne'
    display_name = 'SNNE (Soft Nearest Neighbor Entropy)'
    
    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        """Legacy interface."""
        uncertainty_score, is_hallucination, explanation = SNNE(
            question=question,
            answer=answer,
            samples=samples
        )
        
        # Convert to legacy format
        answer = explanation.get('answer', '')
        samples = explanation.get('samples', [])
        
        return uncertainty_score, answer, samples
