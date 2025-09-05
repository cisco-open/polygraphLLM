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

try:
    from .snne import SNNE
    # For backward compatibility, we also alias TESNNE to SNNE if it doesn't exist as a separate class
    # Since we didn't find TESNNE in the grep search, it's likely the same as SNNE
    TESNNE = SNNE
except ImportError as e:
    import logging
    logging.warning(f"Could not import SNNE due to missing dependencies: {e}")
    SNNE = None
    TESNNE = None
from .kernel_uncertainty import get_entailment_graph, get_semantic_ids_graph
from .p_true import construct_few_shot_prompt, calculate_p_true
from .semantic_entropy import (
    context_entails_response,
    get_semantic_ids_using_entailment,
    get_semantic_ids_using_exact_match,
    get_semantic_ids_using_metric,
    get_semantic_ids_using_embedding,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
    cluster_assignment_entropy,
    weighted_cluster_assignment_entropy,
    soft_nearest_neighbor_loss
)
from .llmUncertainty import LLMUncertainty


# Group functions for easier access
kernel_uncertainty = {
    'get_entailment_graph': get_entailment_graph,
    'get_semantic_ids_graph': get_semantic_ids_graph
}

p_true = {
    'construct_few_shot_prompt': construct_few_shot_prompt,
    'calculate_p_true': calculate_p_true
}

semantic_entropy = {
    'context_entails_response': context_entails_response,
    'get_semantic_ids_using_entailment': get_semantic_ids_using_entailment,
    'get_semantic_ids_using_exact_match': get_semantic_ids_using_exact_match,
    'get_semantic_ids_using_metric': get_semantic_ids_using_metric,
    'get_semantic_ids_using_embedding': get_semantic_ids_using_embedding,
    'logsumexp_by_id': logsumexp_by_id,
    'predictive_entropy': predictive_entropy,
    'predictive_entropy_rao': predictive_entropy_rao,
    'cluster_assignment_entropy': cluster_assignment_entropy,
    'weighted_cluster_assignment_entropy': weighted_cluster_assignment_entropy,
    'soft_nearest_neighbor_loss': soft_nearest_neighbor_loss
}

# Export uncertainty measures
__all__ = [
    'LLMUncertainty',
    'kernel_uncertainty',
    'p_true', 
    'semantic_entropy',
    # Individual functions
    'get_entailment_graph',
    'get_semantic_ids_graph', 
    'construct_few_shot_prompt',
    'calculate_p_true',
    'context_entails_response',
    'get_semantic_ids_using_entailment',
    'get_semantic_ids_using_exact_match',
    'get_semantic_ids_using_metric',
    'get_semantic_ids_using_embedding',
    'logsumexp_by_id',
    'predictive_entropy',
    'predictive_entropy_rao',
    'cluster_assignment_entropy',
    'weighted_cluster_assignment_entropy',
    'soft_nearest_neighbor_loss'
]

# Add SNNE and TESNNE to exports if they were successfully imported
if SNNE is not None:
    __all__.extend(['SNNE', 'TESNNE'])
