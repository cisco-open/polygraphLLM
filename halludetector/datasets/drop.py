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

from datasets import load_dataset

from .parser import Parser


class DropParser(Parser):
    display_name = 'Drop'
    _id = 'drop'

    def __init__(self):
        self.dataset = load_dataset('EleutherAI/drop')
        self.dataset = self.dataset['validation']

    def display(self):
        results = []

        for element in self.dataset:
            answer = ''
            answer_element = element['answer']
            if answer_element['number']:
                answer = answer_element['number']
            elif answer_element['date']['day']:
                answer = f'{answer_element["date"]["year"]}-{answer_element["date"]["month"]}-{answer_element["date"]["day"]}'
            elif answer_element['spans']:
                answer = ' ,'.join(answer_element['spans'])

            samples = []
            for ans in element['validated_answers']['number']:
                if ans:
                    samples.append(ans)
            for ans in element['validated_answers']['date']:
                if ans['day']:
                    samples.append(f'{ans["year"]}-{ans["month"]}-{ans["day"]}')
            for ans in element['validated_answers']['spans']:
                if ans:
                    samples.append(' ,'.join(ans))

            results.append(
                {
                    'question': element['question'],
                    'context': element['passage'],
                    'answer': answer,
                    'samples': samples,
                    'query_id': element['query_id']
                }
            )
        return {
            'data': results,
            'columns': ['question', 'context', 'answer', 'samples', 'query_id']
        }
