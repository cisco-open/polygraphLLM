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
import logging
from polygraphLLM.utils.settings.settings import Settings
logger = logging.getLogger(__name__)


class Detector:

    def __init__(self):
        from polygraphLLM.utils.config import init_config

        config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.json')
        init_config(config_path)

        from polygraphLLM.utils.config import (
            llm_handler, triplets_extractor, sentence_extractor, question_generator,
            retriever, checker, bertscorer, ngramscorer,
        )
        self.llm_handler = llm_handler
        self.triplets_extractor = triplets_extractor
        self.sentence_extractor = sentence_extractor
        self.question_generator = question_generator
        self.retriever = retriever
        self.checker = checker
        self.bertscorer = bertscorer
        self.ngramscorer = ngramscorer
        self.settings_manager = Settings(config_path)

    def ask_llm(self, *args, **kwargs):
        return self.llm_handler.ask_llm(*args, **kwargs)

    def extract_triplets(self, *args, **kwargs):
        return self.triplets_extractor.extract(*args, **kwargs)

    def extract_sentences(self, *args, **kwargs):
        return self.sentence_extractor.extract(*args, **kwargs)

    def generate_question(self, *args, **kwargs):
        return self.question_generator.generate(*args, **kwargs)

    def retrieve(self, *args, **kwargs):
        return self.retriever.retrieve(*args, **kwargs)

    def check(self, *args, **kwargs):
        return self.checker.check(*args, **kwargs)
    
    def find_settings_value(self, settings, search_key):
        return self.settings_manager.get_custom_settings_value(settings, search_key)

    def similarity_bertscore(self, sentences, sampled_passages):
        try:
            return self.bertscorer.predict(
                sentences=sentences,  # list of sentences
                sampled_passages=sampled_passages,  # list of sampled passages
            )
        except Exception as e:
            logger.error(f'Bertscore failed due to {e}')

    def similarity_ngram(self, sentences, passage, sampled_passages):
        return self.ngramscorer.predict(
            passage=passage,
            sentences=sentences,  # list of sentences
            sampled_passages=sampled_passages,  # list of sampled passages
        )

    def score(self, question, answer=None, samples=None, summary=None, settings=None):
        raise NotImplementedError
    
    def detect_hallucination(self, question, answer=None, samples=None, summary=None, settings=None, threshold=0.5):
        """
        Detect hallucination based on threshold. Should be implemented by subclasses.
        
        Args:
            question: Input question
            answer: Model answer (optional, will be generated if not provided)
            samples: Sample responses (optional)
            summary: Summary (optional)
            settings: Configuration settings (optional)
            threshold: Threshold for hallucination detection (0.0 to 1.0)
            
        Returns:
            tuple: (is_hallucinated: bool, raw_score: float, answer: str, additional_data)
        """
        raise NotImplementedError