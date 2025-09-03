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

"""Functions for performance evaluation, mainly used in analyze_results.py."""
from typing import List

import numpy as np
import scipy
from sklearn import metrics
import pandas as pd


# pylint: disable=missing-function-docstring


def normalize(target: List[float]):
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


def skip_nans(target, estimator):
    newt, newe = [], []
    count_nan = 0
    
    for t, e in zip(target, estimator):
        if np.isnan(t) or np.isnan(e):
            count_nan += 1
            continue
        newt.append(t)
        newe.append(e)
    
    return np.array(newt), np.array(newe)


def is_binary_list(lst):
    return all(item in [0, 1] for item in lst)


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def auroc(y_true, y_score):
    y_true, y_score = skip_nans(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def auarc(y_score, y_true):
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    y_true, y_score = skip_nans(y_true, y_score)
    df = pd.DataFrame({"u": y_score, 'a': y_true}).sort_values('u', ascending=True)
    df['amean'] = df['a'].expanding().mean()
    return metrics.auc(np.linspace(0,1,len(df)), df['amean'])


# https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/ue_metrics/pred_rej_area.py
def aucpr(y_score, y_true):
    """
        Measures the area under the Prediction-Rejection curve between `y_score` and `y_true`.

        Parameters:
            y_score (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            y_true (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: area under the Prediction-Rejection curve.
                Higher values indicate better uncertainty estimations.
        """
    y_true, y_score = skip_nans(y_true, y_score)
    y_true = normalize(y_true)
    ue = np.array(y_score)
    num_obs = len(ue)
    num_rej = int(num_obs)
    # Sort in ascending order: the least uncertain come first
    ue_argsort = np.argsort(ue)
    # want sorted_metrics to be increasing => smaller scores is better
    sorted_metrics = np.array(y_true)[ue_argsort]
    # Since we want all plots to coincide when all the data is discarded
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    prr_score = np.sum(scores) / num_rej
    
    return prr_score


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def area_under_thresholded_accuracy(accuracies, uncertainties):
    accuracies, uncertainties = skip_nans(accuracies, uncertainties)
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array([accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles])
    dx = quantiles[1] - quantiles[0]
    area = (select_accuracies * dx).sum()
    return area


# Need wrappers because scipy expects 1D data.
def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func
