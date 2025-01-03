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
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MistralHandler:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def ask_llm(self, prompt, n=1, temperature=0, max_new_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 400))):
        model_inputs = self.tokenizer([prompt] * n, return_tensors="pt")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
        results = [r for r in self.tokenizer.batch_decode(generated_ids)]
        logger.info(f'Prompt responses: {results}')
        return results
