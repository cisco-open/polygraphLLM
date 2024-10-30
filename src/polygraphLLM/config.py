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

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram

from polygraphLLM.llm.openai import OpenAIHandler
from polygraphLLM.llm.mistral import MistralHandler
from polygraphLLM.extractor.extractor import TripletsExtractorHandler, SentenceExtractorHandler
from polygraphLLM.generators.question import QuestionGenerator
from polygraphLLM.retrievers.retriever import RetrieverHandler
from polygraphLLM.checker.checker import CheckerHandler

llm_handler = None
triplets_extractor = None
sentence_extractor = None
question_generator = None
retriever = None
checker = None
bertscorer = None
ngramscorer = None


def init_config(file, force=False):
    try:
        with open(file, 'r') as cfgfile:
            config = json.loads(cfgfile.read())
            for c in config:
                key = c['key']
                value = c['value']
                if force or not os.getenv(key):
                    os.environ[key] = str(value)
    except Exception:
        print('No config file. Loading variables from environment.')
    init_building_blocks()


def init_building_blocks(force=False):
    global llm_handler, triplets_extractor, sentence_extractor, question_generator, \
        retriever, checker, bertscorer, ngramscorer

    if llm_handler is None or force:
        llm_handler = OpenAIHandler()
        triplets_extractor = TripletsExtractorHandler(llm_handler)
        sentence_extractor = SentenceExtractorHandler()
        question_generator = QuestionGenerator(llm_handler)
        retriever = RetrieverHandler(sentence_extractor)
        checker = CheckerHandler(sentence_extractor, llm_handler)
        bertscorer = SelfCheckBERTScore(rescale_with_baseline=True)
        ngramscorer = SelfCheckNgram(n=1)
