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
Kernel Uncertainty: Graph-based uncertainty estimation using entailment relationships.

This algorithm constructs a graph of entailment relationships between generated
responses and uses graph properties to estimate uncertainty.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, List

from ...utils.base_detector import BaseDetector

# Import kernel uncertainty implementation from the original location
try:
    from ....polygraphLLM.detectors.snne.uncertainty.uncertainty_measures.kernel_uncertainty import (
        get_entailment_graph,
        get_semantic_ids_graph,
        EntailmentDeberta
    )
    import networkx as nx
    HAS_KERNEL_UNCERTAINTY = True
except ImportError:
    logger.warning("Kernel uncertainty implementation not available")
    HAS_KERNEL_UNCERTAINTY = False

logger = logging.getLogger(__name__)


def kernel_uncertainty(
    question: str,
    answer: str = None,
    context: str = None,
    samples: List[str] = None,
    threshold: float = 0.5,
    n_samples: int = 5,
    temperature: float = 0.8,
    is_weighted: bool = True,
    weight_strategy: str = "manual",
    **kwargs
) -> Tuple[float, bool, Dict]:
    """
    Estimate uncertainty using kernel (graph-based) methods.
    
    Args:
        question: The input question/query
        answer: The answer to analyze (if None, generates one)
        context: Optional context (unused)
        samples: Pre-generated samples (if None, generates them)
        threshold: Uncertainty threshold for hallucination decision (0.0-1.0)
        n_samples: Number of samples to generate
        temperature: Temperature for sample generation
        is_weighted: Whether to use weighted edges in the graph
        weight_strategy: Strategy for edge weighting ("manual", "deberta")
        **kwargs: Additional parameters
        
    Returns:
        uncertainty_score (float): Graph-based uncertainty measure
        is_hallucination (bool): True if uncertainty exceeds threshold
        explanation (dict): Contains graph analysis details
    """
    detector = BaseDetector()
    
    if not HAS_KERNEL_UNCERTAINTY:
        logger.error("Kernel uncertainty implementation not available")
        explanation = {
            'algorithm': 'kernel_uncertainty',
            'error': 'Kernel uncertainty implementation not available'
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
            logger.warning("Not enough samples for kernel uncertainty computation")
            explanation = {
                'algorithm': 'kernel_uncertainty',
                'error': 'Not enough samples',
                'answer': answer,
                'question': question
            }
            return 0.5, False, explanation
        
        # Compute kernel uncertainty
        uncertainty_score, graph_info = _compute_kernel_uncertainty(
            samples,
            is_weighted=is_weighted,
            weight_strategy=weight_strategy
        )
        
        # Prepare explanation
        explanation = {
            'algorithm': 'kernel_uncertainty',
            'uncertainty_score': uncertainty_score,
            'threshold': threshold,
            'n_samples': len(samples),
            'temperature': temperature,
            'is_weighted': is_weighted,
            'weight_strategy': weight_strategy,
            'graph_info': graph_info,
            'samples': samples,
            'answer': answer,
            'question': question
        }
        
        # Make binary decision
        is_hallucination = uncertainty_score >= threshold
        
        logger.info(f'Kernel Uncertainty: uncertainty={uncertainty_score:.3f}, threshold={threshold}, hallucination={is_hallucination}')
        
        return uncertainty_score, is_hallucination, explanation
        
    except Exception as e:
        logger.error(f'Kernel uncertainty failed: {e}')
        explanation = {
            'algorithm': 'kernel_uncertainty',
            'error': str(e),
            'answer': answer or "Error generating answer"
        }
        return 0.5, False, explanation


def _compute_kernel_uncertainty(
    generations: List[str],
    is_weighted: bool = True,
    weight_strategy: str = "manual"
) -> Tuple[float, Dict]:
    """Compute kernel uncertainty using graph properties."""
    try:
        # Initialize entailment model
        model = EntailmentDeberta()
        
        # Construct entailment graph
        graph = get_entailment_graph(
            generations,
            model,
            is_weighted=is_weighted,
            weight_strategy=weight_strategy
        )
        
        # Compute graph properties for uncertainty estimation
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        
        # Basic graph metrics
        density = nx.density(graph)
        
        # Connected components analysis
        connected_components = list(nx.connected_components(graph))
        n_components = len(connected_components)
        
        # Compute uncertainty based on graph structure
        # Higher connectivity -> lower uncertainty
        # More components -> higher uncertainty
        if n_nodes == 0:
            uncertainty = 0.5
        else:
            # Normalize component count by total nodes
            component_ratio = n_components / n_nodes
            
            # Combine density and component measures
            # Low density or high component ratio indicates high uncertainty
            uncertainty = (1.0 - density) * 0.5 + component_ratio * 0.5
            uncertainty = min(max(uncertainty, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Graph analysis info
        graph_info = {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'n_components': n_components,
            'component_sizes': [len(comp) for comp in connected_components],
            'is_weighted': is_weighted,
            'weight_strategy': weight_strategy
        }
        
        return uncertainty, graph_info
        
    except Exception as e:
        logger.error(f"Kernel uncertainty computation failed: {e}")
        return 0.5, {'error': str(e)}  # Return neutral uncertainty on failure
