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

"""Implement semantic entropy."""
import os
import pickle
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

from ..models.huggingface_models import HuggingfaceModel
from ..utils.utils import save, md5hash
from ..utils.entropy_utils import greedy_clustering, dfs_clustering, snne, entailment_similarity_matrix


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEmbedding:
    def __init__(self):
        pass
    
    def get_similarity_score(self, list_text):
        list_embedding = self.model.encode(list_text, normalize_embeddings=True)
        
        return self.model.similarity(list_embedding, list_embedding)


class Qwen2Embedding(BaseEmbedding):
    def __init__(self):
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
        self.model.max_seq_length = 8192
    

class SFR2Embedding(BaseEmbedding):
    def __init__(self):
        self.model = SentenceTransformer("Salesforce/SFR-Embedding-2_R", trust_remote_code=True)


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info(f'Deberta Input: {text1} -> {text2}')
            logging.info(f'Deberta Prediction: {prediction}')

        return prediction
    
    def get_similarity_score(self, text1, text2, strict_entailment=True, exclude_neutral=True):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        # Take the softmax score for the `entailment` class
        softmax_logits = F.softmax(logits, dim=1)
        if strict_entailment:
            prediction = softmax_logits[:, 2].cpu().item()
        elif exclude_neutral:
            # LUQ paper
            prediction = (softmax_logits[:, 2] / (softmax_logits[:, 2] + softmax_logits[:, 0])).cpu().item()
        else:
            # w = (0, 0.5, 1) as in KLE's paper
            prediction = (softmax_logits[:, 2] + softmax_logits[:, 1] * 0.5).cpu().item()

        return prediction


class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info(f'Restoring prediction cache from {entailment_cache_id}')

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['question'])

        logging.info(f'{self.name} input: {prompt}')

        hashed = md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info(f'{self.name} prediction: {response}')

        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('MANUAL NEUTRAL!')
            return 1


class EntailmentLlama(EntailmentLLM):

    def __init__(self, entailment_cache_id, entailment_cache_only, name):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = name
        self.model = HuggingfaceModel(
            name, stop_sequences='default', max_new_tokens=30)

    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
        prompt += "Response:"""

        return prompt

    def predict(self, prompt, temperature):
        predicted_answer, _, _ = self.model.predict(prompt, temperature)
        return predicted_answer


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids_using_entailment(strings_list, model, strict_entailment=False, cluster_method='greedy', example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(i, j):
        text1 = strings_list[i]
        text2 = strings_list[j]

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    if cluster_method == 'dfs':
        semantic_set_ids = dfs_clustering(strings_list, are_equivalent)
    elif cluster_method == 'greedy':
        semantic_set_ids = greedy_clustering(strings_list, are_equivalent)

    return semantic_set_ids


def get_semantic_ids_using_exact_match(strings_list, cluster_method='greedy'):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(i, j):
        semantically_equivalent = 1 if strings_list[i] == strings_list[j] else 0

        return semantically_equivalent

    if cluster_method == 'dfs':
        semantic_set_ids = dfs_clustering(strings_list, are_equivalent)
    elif cluster_method == 'greedy':
        semantic_set_ids = greedy_clustering(strings_list, are_equivalent)

    return semantic_set_ids


def get_semantic_ids_using_metric(strings_list, metric, example, cluster_method='greedy'):
    """Group list of predictions into semantic meaning."""
    # TODO: squad metric, i.e. F1, is not symmetric

    def are_equivalent(i, j):
        example['answers']['text'] = [strings_list[j]]
        semantically_equivalent = metric(strings_list[i], example)

        return semantically_equivalent

    if cluster_method == 'dfs':
        semantic_set_ids = dfs_clustering(strings_list, are_equivalent)
    elif cluster_method == 'greedy':
        semantic_set_ids = greedy_clustering(strings_list, are_equivalent)

    return semantic_set_ids


def get_semantic_ids_using_embedding(strings_list, model, cluster_method='dfs', threshold=0.5):
    """Group list of predictions into semantic meaning."""

    similarity_matrix = model.get_similarity_score(strings_list)
    
    def are_equivalent(i, j):
        return similarity_matrix[i, j] > threshold
        
    if cluster_method == 'dfs':
        semantic_set_ids = dfs_clustering(strings_list, are_equivalent)
    elif cluster_method == 'greedy':
        semantic_set_ids = greedy_clustering(strings_list, are_equivalent)

    return semantic_set_ids, similarity_matrix


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized', return_unique_ids=False):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    if return_unique_ids:
        return unique_ids, log_likelihood_per_semantic_id
    
    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def weighted_cluster_assignment_entropy(semantic_ids, log_probs):
    """Use token probability to weight cluster assignment entropy"""
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (np.exp(log_probs) * np.log(probabilities)).sum()
    return entropy


def soft_nearest_neighbor_loss(strings_list, entailment_model, embedding_model, semantic_ids, similarity_matrix=None, variant="only_denom", similarity_model="entailment", temperature=1.0, exclude_diagonal=True, strict_entailment=True, weight=None):
    if similarity_matrix is None:
        if similarity_model == "entailment":
            similarity_matrix = entailment_similarity_matrix(
                entailment_model, 
                strings_list, 
                strict_entailment=strict_entailment)
        elif similarity_model == "embedding":
            similarity_matrix = embedding_model.get_similarity_score(strings_list)

    snn_loss = snne(similarity_matrix, semantic_ids, variant=variant, temperature=temperature, exclude_diagonal=exclude_diagonal, weight=weight)
    
    return snn_loss
