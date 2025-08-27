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
from sklearn.cluster import KMeans


def get_D_mat(W):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    
    return D

def get_L_mat(W, symmetric=True):
    # compute the degree matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
        
    return L.copy()

def get_eig(L, thres=None, eps=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    eigvals, eigvecs = np.linalg.eigh(L)

    #eigvals, eigvecs = np.linalg.eig(L)
    #assert np.max(np.abs(eigvals.imag)) < 1e-5
    #eigvals = eigvals.real
    #idx = eigvals.argsort()
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:,idx]

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
        
    return eigvals, eigvecs

def find_equidist(P, eps=1e-4):
    from scipy.linalg import eig
    P = P / P.sum(1)[:, None]
    P = (1-eps) * P + eps * np.eye(len(P))
    assert np.abs(P.sum(1)-1).max() < 1e-3
    w, vl, _ = eig(P, left=True)
    #assert np.max(np.abs(w.imag)) < 1e-5
    w = w.real
    idx = w.argsort()
    w = w[idx]
    vl = vl[:, idx]
    assert np.max(vl[:, -1].imag) < 1e-5
    
    return vl[:, -1].real / vl[:, -1].real.sum()

class SpetralClusteringFromLogits:
    def __init__(self,
                 eigv_threshold=0.9,
                 cluster=True,
                 adjust=False) -> None:
        self.eigv_threshold = eigv_threshold
        self.rs = 0
        self.cluster = cluster
        self.adjust = adjust

    def get_laplacian(self, W):
        L = get_L_mat(W, symmetric=True)
        
        return L

    def get_eigvs(self, W):
        L = self.get_laplacian(W)
        
        return (1 - get_eig(L)[0])

    def __call__(self, W, cluster=None):
        if cluster is None: cluster = self.cluster
        L = self.get_laplacian(W)
        if not cluster:
            return (1-get_eig(L)[0]).clip(0 if self.adjust else -1).sum()
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        
        return kmeans.labels_

    def eig_entropy(self, W):
        L = get_L_mat(W, symmetric=True)
        eigs = get_eig(L, eps=1e-4)[0] / W.shape[0]
        
        return np.exp(- (eigs * np.nan_to_num(np.log(eigs))).sum())

    def proj(self, W):
        L = get_L_mat(W, symmetric=True)
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        
        return eigvecs

    def kmeans(self, eigvecs):
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        
        return kmeans.labels_