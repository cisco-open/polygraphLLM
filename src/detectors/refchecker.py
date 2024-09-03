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

from .base import Detector


class RefChecker(Detector):
    id = 'refchecker'
    display_name = 'RefChecker'

    def __init__(self):
        super().__init__()

    def score(self, question=None, answer=None, samples=None,summary=None, settings=None):
        if not question:
            question = self.generate_question(answer)

        if not answer:
            answer = self.ask_llm(question.strip())[0]

        triplets = self.extract_triplets(answer, question, max_new_tokens=200)
        reference = self.retrieve([question])

        results = [
            self.check(t, reference, question=question)
            for t in triplets
        ]
        agg_results = self.soft_agg(results)
        for k, v in agg_results.items():
            agg_results[k] = float("{:.2f}".format(v))
        return agg_results, answer, None

    def soft_agg(self, results):
        """Aggregate results by taking the ratio of each category."""
        if not results:
            return {
                "Entailment": 0.0,
                "Neutral": 0.0,
                "Contradiction": 0.0,
                "Abstain": 1.0,
            }
        total = len(results)
        agg = {
            "Entailment": 0.0,
            "Neutral": 0.0,
            "Contradiction": 0.0,
            "Abstain": 0.0,
        }
        for result in results:
            agg[result] += 1.0
        for key in agg:
            agg[key] /= total
        print(results)
        return agg
