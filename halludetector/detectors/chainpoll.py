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
import os

from .base import Detector
from ..prompts.default import DEFAULT_CHAINPOLL_PROMPT

logger = logging.getLogger(__name__)


class ChainPoll(Detector):
    id = 'chainpoll'
    display_name = 'Chain Poll'

    def __init__(self):
        super().__init__()
        self.sample_number = int(os.getenv("CHAINPOLL_SAMPLING_NUMBER", 5))
        try:
            with open(f'{os.path.dirname(os.path.realpath(__file__))}/../prompts/chainpoll.txt', 'r') as pf:
                self.prompt = pf.read()
        except:
            self.prompt = DEFAULT_CHAINPOLL_PROMPT

    def check_hallucinations(self, completion, question):
        text = self.prompt.format(completion=completion, question=question)
        responses = self.ask_llm(text, n=self.sample_number, temperature=0.2)
        logger.info(f'Hallucination check response: {responses}')
        return [response.lower().startswith("yes") for response in responses], responses

    def score(self, question, answer=None, samples=None, summary=None):
        if not answer:
            answer = self.ask_llm(question.strip())[0]
        hallucinations, responses = self.check_hallucinations(answer.strip(), question.strip())

        score = hallucinations.count(True) / len(hallucinations)
        return score, answer, responses
