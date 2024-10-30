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

import logging

from polygraphLLM.detectors import (
    ChainPoll, SelfCheckGPTBertScore,
    SelfCheckGPTNGram, SelfCheckGPTMQAG,
    RefChecker, GEval,
    SelfCheckGPTPrompt
)

scorer_mapping = {
    'self_check_gpt_bertscore': SelfCheckGPTBertScore,
    'self_check_gpt_ngram': SelfCheckGPTNGram,
    'self_check_gpt_prompt': SelfCheckGPTPrompt,
    'refchecker': RefChecker,
    'g_eval': GEval,
    'chainpoll': ChainPoll,
}

logger = logging.getLogger(__name__)


def calculate_score(method, question, answer=None, samples=None, summary=None):
    scorer_class = scorer_mapping.get(method)
    if scorer_class:
        scorer = scorer_class()
    return scorer.score(question, answer, samples, summary)
