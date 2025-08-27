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

import numpy as np
import torch

from . import clustering as pc


def greedy_clustering(strings_list, are_equivalent):
    # Initialise all ids with -1.
    N = len(strings_list)
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i in range(N):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, N):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(i, j):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def dfs_clustering(strings_list, are_equivalent):
    N = len(strings_list)
    visited = [False] * N
    semantic_set_ids = [-1] * N
    current_group = 0
    
    def dfs(node):
        stack = [node]
        while stack:
            v = stack.pop()
            for neighbor in range(N):
                # if similarity_matrix[v, neighbor] > threshold and not visited[neighbor]:
                if are_equivalent(v, neighbor) and not visited[neighbor]:
                    visited[neighbor] = True
                    semantic_set_ids[neighbor] = current_group
                    stack.append(neighbor)
    
    for i in range(N):
        if not visited[i]:
            visited[i] = True
            semantic_set_ids[i] = current_group
            dfs(i)
            current_group += 1
            
    return semantic_set_ids


def snne(similarity_matrix, labels, variant="only_denom", temperature=1.0, epsilon=1e-8, exclude_diagonal=True, weight=None):
    # Convert inputs to tensors if they are not already
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)
    
    # Ensure the labels are a column vector for broadcasting
    labels = labels.view(-1, 1)

    # Create a mask for the labels to identify dissimilar pairs
    label_mask = labels != labels.T
    label_inf = torch.zeros_like(similarity_matrix)
    label_inf[label_mask] = float('-inf')

    # Divide the similarity matrix by temperature
    similarity_matrix = similarity_matrix / temperature
    
    if exclude_diagonal:
        # Discard self similarity
        diag_inf = torch.diag(torch.tensor(float('-inf')).expand(labels.size(0)))
        similarity_matrix = similarity_matrix + diag_inf
    
    # Use log-sum-exp trick to stabilize the computation
    # when temperature is very low
    logsumexp_numerators = torch.logsumexp(similarity_matrix + label_inf, dim=1, keepdim=True)
    logsumexp_denominators = torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
    
    # Replace -inf in numerators with log(epsilon)
    # when a class has only one sample
    inf_mask = torch.isinf(logsumexp_numerators)
    logsumexp_numerators[inf_mask] = torch.log(torch.tensor(epsilon))

    # Calculate the loss
    if variant == "full":
        loss = logsumexp_numerators - logsumexp_denominators
    elif variant == "only_num":
        loss = logsumexp_numerators
    elif variant == "only_denom":
        loss = logsumexp_denominators
    elif variant == "num_minus_denom":
        loss = 2 * torch.exp(logsumexp_numerators) - torch.exp(logsumexp_denominators) + torch.exp(torch.tensor(1./temperature)) * logsumexp_numerators.size(0)
        loss = torch.log(loss)
        
    # Weighted loss
    if weight is None:
        weight = torch.ones_like(loss)
    elif weight.size() != loss.size():
        weight = weight.view(-1, 1)
        
    loss = -(loss * weight).mean()
    
    return loss


def entailment_similarity_score(entailment_model, text1, text2, strict_entailment=True, exclude_neutral=True, bidirectional=True):
    score_1 = entailment_model.get_similarity_score(
        text1, 
        text2, 
        strict_entailment=strict_entailment,
        exclude_neutral=exclude_neutral,
    )
    if bidirectional:
        score_2 = entailment_model.get_similarity_score(
            text2, 
            text1, 
            strict_entailment=strict_entailment,
            exclude_neutral=exclude_neutral
        )
        
        return (score_1 + score_2) / 2
    else:
        return score_1
    
def entailment_similarity_matrix(entailment_model, list_strings, strict_entailment=True, exclude_neutral=True, bidirectional=True):
    n_samples = len(list_strings)
    similarity_matrix = torch.eye(n_samples)
    
    for i in range(n_samples - 1):
        for j in range(i + 1, n_samples):
            similarity = entailment_similarity_score(
                entailment_model, 
                list_strings[i], 
                list_strings[j], 
                strict_entailment=strict_entailment,
                exclude_neutral=exclude_neutral,
                bidirectional=bidirectional
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix


def lexical_similarity_matrix(rouge, list_strings, tokenizer=None):
    n_samples = len(list_strings)
    similarity_matrix = torch.eye(n_samples)
    
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            similarity = rouge.compute(
                predictions=[list_strings[i]], 
                references=[list_strings[j]], 
                rouge_types=['rougeL'], 
                tokenizer=tokenizer)['rougeL']
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix


def exact_match_similarity_matrix(list_strings):
    n_samples = len(list_strings)
    similarity_matrix = torch.eye(n_samples)
    
    for i in range(n_samples-1):
        for j in range(i+1, n_samples):
            similarity = 1. if list_strings[i] == list_strings[j] else 0.
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix


def squad_f1_similarity_matrix(squad_f1, list_strings, example, symmetric=False):
    n_samples = len(list_strings)
    similarity_matrix = torch.eye(n_samples)
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                example['answers']['text'] = [list_strings[j]]
                similarity = squad_f1(list_strings[i], example)
                similarity_matrix[i][j] = similarity
    
    if symmetric:
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    
    return similarity_matrix


def get_tokenwise_importance(measure_model, tokenizer, generations, question):
    token_importance_list = []
    
    for k in range(len(generations)):
        generated_text = generations[k]
        tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))

        # likelihood = likelihoods[k]['original_token_wise_entropy']
        token_importance = []
        # measure cosine similarity by removing each token and compare the similarity
        for token in tokenized:
            similarity_to_original = measure_model.predict([
                    question + generated_text,
                    question + generated_text.replace(tokenizer.decode(token, skip_special_tokens=True), '')
            ], show_progress_bar=False)
            token_importance.append(1 - torch.tensor(similarity_to_original))

        token_importance = torch.tensor(token_importance).reshape(-1)
        token_importance_list.append(token_importance)
    
    return token_importance_list


def get_sentence_similarites(measure_model, generations_with_question):
    n_samples = len(generations_with_question)
    similarity_matrix = np.eye(n_samples)

    for i in range(len(generations_with_question)):
        for j in range(i+1, len(generations_with_question)):
            similarity = measure_model.predict([
                generations_with_question[i], 
                generations_with_question[j]
            ], show_progress_bar=False)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix


def compute_lexical_similarity(sim_mat):
    n_samples = len(sim_mat)
    num = 0.
    denom = 0
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                num += sim_mat[i,j]
                denom += 1
    
    if denom == 0: return 1.
    
    return num / denom


def get_spectral_eigv(similarity_matrix, adjust=True):
    clusterer = pc.SpetralClusteringFromLogits(
        eigv_threshold=None,
        cluster=False)
    
    return clusterer.get_eigvs(similarity_matrix).clip(0 if adjust else -1).sum()


def get_degreeuq(similarity_matrix):
    ret = np.asarray(np.sum(1 - similarity_matrix, axis=1))
    
    return ret.mean(), ret


def get_luq_pair(similarity_matrix):
    # S[i, j]: i -> j
    # Exclude self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    ret = np.asarray(1 - np.max(similarity_matrix, axis=1))
    
    return ret.mean(), ret


def get_eccentricity(similarity_matrix, eigv_threshold=0.9):
    clusterer = pc.SpetralClusteringFromLogits(
        eigv_threshold=eigv_threshold,
        cluster=False)
    projected = clusterer.proj(similarity_matrix)
    
    ds = np.linalg.norm(projected - projected.mean(0)[None, :], 2, axis=1)
    
    return np.linalg.norm(ds, 2, axis=0), ds


def get_sar(list_token_importance, list_sentence_similarity_matrix, list_token_log_likelihoods, temp=0.001):
    sar = []
    
    for token_importance, sentence_similarity_matrix, token_log_likelihoods in zip(list_token_importance, list_sentence_similarity_matrix, list_token_log_likelihoods):
        token_sar = []
        
        for importance, log_likelihoods in zip(token_importance, token_log_likelihoods):
            num_tokens = min(len(importance), len(log_likelihoods))
            log_likelihoods = np.array(log_likelihoods)[:num_tokens]
            importance = np.array(importance)[:num_tokens]
            R_t = 1 - importance
            R_t_norm = R_t / R_t.sum()
            E_t = -log_likelihoods * R_t_norm
            token_sar.append(E_t.sum())
        
        token_sar = np.array(token_sar)
        probs_token_sar = np.exp(-token_sar)
        R_s = (
            probs_token_sar
            * sentence_similarity_matrix
            * (1 - np.eye(sentence_similarity_matrix.shape[0]))
        )
        sent_relevance = R_s.sum(-1) / temp
        E_s = -np.log(sent_relevance + probs_token_sar)
        sar.append(E_s.mean())
        
    return np.array(sar)


def get_eigenscore(list_embeddings, alpha=0.001):
    sentence_embeddings = np.array(list_embeddings)
    dim = sentence_embeddings.shape[-1]
    J_d = np.eye(dim) - 1 / dim * np.ones((dim, dim))
    covariance = sentence_embeddings @ J_d @ sentence_embeddings.T
    reg_covariance = covariance + alpha * np.eye(covariance.shape[0])
    eigenvalues = np.linalg.eigvalsh(reg_covariance)
    
    return np.mean(np.log([val if val > 0 else 1e-10 for val in eigenvalues]))