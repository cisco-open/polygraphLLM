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

import os
import pickle
import logging
from collections import defaultdict

import wandb
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    entailment_file = 'deberta_entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def check_implication(self, text1, text2, *args, **kwargs):
        hashed = utils.md5hash(f"Text1 for DeBerta: {text1}, Text2 for DeBerta: {text2}")
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            prediction, confidence = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
            # The model checks if text1 -> text2, i.e. if text2 follows from text1.
            # check_implication('The weather is good', 'The weather is good and I like you') --> 1
            # check_implication('The weather is good and I like you', 'The weather is good') --> 2

            outputs = self.model(**inputs)
            logits = outputs.logits
            # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
            activations = F.softmax(logits, dim=1)
            largest_index = torch.argmax(activations)  # pylint: disable=no-member
            confidence = torch.max(activations)
            prediction = largest_index.cpu().item()
            if os.environ.get('DEBERTA_FULL_LOG', False):
                logging.info('Deberta Input: %s -> %s', text1, text2)
                logging.info('Deberta Prediction: %s', prediction)
                logging.info('Deberta Prediction Prob: %s', confidence)
            self.prediction_cache[hashed] = (prediction, confidence.cpu().item())
        return prediction, confidence
    
    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        utils.save(self.prediction_cache, self.entailment_file)


def get_entailment_graph(strings_list, model, is_weighted=False, example=None, weight_strategy="manual"):
    """
    Get graph of entailment
    """
    def get_edge(text1, text2, is_weighted=False, example=None):
        implication_1, prob_impl1 = model.check_implication(text1, text2, example=example)
        implication_2, prob_impl2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2])
        weight = int(implication_1 == 2) + int(implication_2 == 2) + 0.5 * int(implication_1 == 1) + 0.5 * int(implication_2 == 1)
        if is_weighted:
            if weight_strategy == "manual":
                return weight
            elif weight_strategy == "deberta":
                return prob_impl1 + prob_impl2
            else:
                raise ValueError(f"Unknown weight strategy {weight_strategy}")
        return weight >= 1.5

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    nodes = range(len(strings_list))
    edges = []
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                edge = get_edge(string1, strings_list[j], example=example, is_weighted=is_weighted)
                if is_weighted:
                    if edge:
                        edges.append((i, j, edge))
                else:
                    edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    return G


def get_semantic_ids_graph(strings_list, model, semantic_ids, ordered_ids, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_similar(text1, text2):

        implication_1, prob = model.check_implication(text1, text2, example=example)
        implication_2, prob = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        return (implication_1 == 2) + (implication_1 == 1) * 0.5 +\
               (implication_2 == 2) + (implication_2 == 1) * 0.5

    # Initialise all ids with -1.
    nodes = ordered_ids
    weights = defaultdict(list) # (i, j) -> weight
    for i, string1 in enumerate(strings_list):
        node_i = semantic_ids[i]
        for j in range(i + 1, len(strings_list)):
            node_j = semantic_ids[j]
            edge_weight = are_similar(string1, strings_list[j])
            if edge_weight > 0:
                weights[(node_i, node_j)].append(edge_weight)
    for k, v in weights.items():
        weights[k] = np.sum(v)
    assert -1 not in semantic_ids
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(i, j, w) for (i, j), w in weights.items()])
    return G