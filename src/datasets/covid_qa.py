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

import json
import os

from .parser import Parser


class CovidQAParser(Parser):
    display_name = 'Covid-QA'
    _id = 'covid-qa'

    def __init__(self, file='../../datasets/Covid-QA.json'):
        self.file = file
        if self.file.startswith('..'):
            self.file = f'{os.path.dirname(os.path.realpath(__file__))}/{file}'
        with open(self.file, 'r') as parsefile:
            self.data = json.loads(parsefile.read())

    def display(self):
        results = []
        for data in self.data['data']:
            for paragraph in data['paragraphs']:
                for qas in paragraph['qas']:
                    results.append({'question': qas['question'], 'id': qas['id'], 'answer': qas['answers'][0]['text']})
        return {
            'data': results,
            'columns': ['id', 'question', 'answer']
        }
