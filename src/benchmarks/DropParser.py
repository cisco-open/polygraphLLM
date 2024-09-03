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
import uuid
from .DatasetParser import DatasetParser

class DropParser(DatasetParser):
    display_name = 'Drop'
    id = 'drop'

    def download_data(self):
        self.dataset = load_dataset('EleutherAI/drop', trust_remote_code=True)
        self.dataset = self.dataset['train']


    def display(self, offset, limit):
        result = []
        for item in self.dataset:
            answer_spans = item["answer"]["spans"]
            answer_number = item["answer"]["number"]
            if not answer_spans and not answer_number:
                answer = item["answer"]["date"]["year"]
            else:
                if answer_spans:
                    answer = ", ".join(answer_spans)
                else:
                    answer = answer_number

            result.append({
                "id": uuid.uuid4(),
                "question": item["question"],
                "answer": answer,
                "context": item["passage"],
            })
        return self.apply_offset_limit(result, offset, limit)
