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

import re

from .base import Detector

from ..utils.prompts.default import DEFAULT_COH_PROMPT, DEFAULT_FLU_PROMPT, DEFAUL_REL_PROMPT, DEFAULT_CON_PROMPT

logger = logging.getLogger(__name__)


class GEval(Detector):
    id = 'g_eval'
    display_name = 'G-Eval'
    metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    
    @staticmethod
    def parse_output(output):
        if ':' in output:
            output = output.rsplit(':', 1)[-1]
        matched = re.search(r"^ ?([\d\.]+)", output)
        if matched:
            try:
                score = float(matched.group(1))
            except:
                score = 0
        else:
            if ':' in output:
                output = output.rsplit(':', 1)[-1]
                matched = re.search(r"^ ?([\d\.]+)", output)
                if matched:
                    try:
                        score = float(matched.group(1))
                    except:
                        score = 0
            else:
                score = 0
        return score

    @staticmethod
    def normalize_score(score):
        max_score = 5  # Maximum possible score
        normalized_score = score / max_score
        return normalized_score

    def create_prompt(self, answer, summary, metric):
        mapping = {
            "coherence": ("coh_detailed.txt", DEFAULT_COH_PROMPT),
            "consistency": ("con_detailed.txt", DEFAULT_CON_PROMPT),
            "fluency": ("flu_detailed.txt", DEFAULT_FLU_PROMPT),
            "relevance": ("rel_detailed.txt", DEFAUL_REL_PROMPT)
        }
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + f'/../prompts/{mapping[metric][0]}') as readfile:
                prompt = readfile.read()
        except:
            prompt = mapping[metric][1]

        cur_prompt = prompt.replace('{{Document}}', answer).replace('{{Summary}}', summary)
        return cur_prompt

    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        scores = {}
        samples = []
        if not answer:
            answer = self.ask_llm(question)[0]
        answer = answer.strip()
        if not summary:
            summary_prompt = f"Create a summary with 20 maximum words from {answer}"
            summary = self.ask_llm(summary_prompt)[0].strip()
        for metric in self.metrics:
            prompt = self.create_prompt(answer, summary, metric)
            n = self.find_settings_value(settings, "GEVAL_SAMPLING_NUMBER")
            answers = self.ask_llm(prompt, n)
            samples.append(answers)
            all_scores = [self.parse_output(x.strip()) for x in answers]
            score = sum(all_scores) / len(all_scores)
            scores[metric.title()] = float("{:.2f}".format(self.normalize_score(score)))
        scores['Total'] = float("{:.2f}".format(sum([v for k, v in scores.items()])/len(scores)))
        return scores, answer, samples
    
    def detect_hallucination(self, question, answer=None, samples=None, summary=None, settings=None, threshold=0.5):
        """
        Detect hallucination based on threshold. Lower total score indicates hallucination.
        
        Returns:
            tuple: (is_hallucinated: bool, raw_score: float, answer: str, additional_data)
        """
        scores, answer, samples = self.score(question, answer, samples, summary, settings)
        total_score = scores.get('Total', 0.0)
        # Lower quality score indicates higher chance of hallucination
        is_hallucinated = bool(total_score < threshold)
        return is_hallucinated, total_score, answer, scores