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

import spacy

from .gpt4_extractor import GPT4Extractor


class TripletsExtractorHandler:
    def __init__(self, llm_handler):
        self.extractor = GPT4Extractor(llm_handler)

    def extract(self, response, question=None, max_new_tokens=200):
        return self.extractor.extract_claim_triplets(response, question, max_new_tokens)


class SentenceExtractorHandler:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text):
        return [sent for sent in self.nlp(text).sents]
