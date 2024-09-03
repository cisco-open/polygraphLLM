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

import os
import json
import uuid
from .DatasetParser import DatasetParser

class CovidQAParser(DatasetParser):
    display_name = 'Covid QA'
    id = 'covid-qa'

    def __init__(self, file='../datasets/Covid-QA.json'):
        self.file = file
        if self.file.startswith('..'):
            self.file = f'{os.path.dirname(os.path.realpath(__file__))}/{file}'
        with open(self.file, 'r') as parsefile:
            self.dataset = json.loads(parsefile.read())

    def display(self, offset, limit):
        result = []
        for item in self.dataset["data"]:
            for qa in item["paragraphs"][0]["qas"]:
                result.append({
                    "id": uuid.uuid4(),
                    "question": qa["question"],
                    "answer": qa["answers"][0]["text"],
                    "context": item["paragraphs"][0]["context"],
                })
        return self.apply_offset_limit(result, offset, limit)
    