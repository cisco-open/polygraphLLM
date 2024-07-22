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

import requests

from .parser import Parser


class SummEval(Parser):
    display_name = 'SummEval'
    _id = 'summeval'
    url = 'https://raw.githubusercontent.com/nlpyang/geval/main/data/summeval.json'

    def __init__(self):
        response = requests.get(self.url)
        self.dataset = response.json()

    def display(self):
        results = []

        for element in self.dataset:
            results.append(
                {
                    'answer': element['source'],
                    'summary': element['system_output']
                }
            )
        return {
            'data': results,
            'columns': ['answer', 'summary']
        }
