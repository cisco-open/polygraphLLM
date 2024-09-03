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


class DollyParser(Parser):
    display_name = 'Databricks Dolly'
    _id = 'databricks-dolly'

    def __init__(self):
        self.dataset = load_dataset('databricks/databricks-dolly-15k')
        self.dataset = self.dataset['train']

    def display(self):
        results = []

        for element in self.dataset:
            results.append(
                {
                    'question': element['instruction'],
                    'context': element['context'],
                    'answer': element['response'],
                    'category': element['category']
                }
            )
        return {
            'data': results,
            'columns': ['question', 'context', 'answer', 'category']
        }
