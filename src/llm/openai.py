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
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def ask_llm(self, prompt, n=1, temperature=0.5, max_new_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 400))):
        response = self.client.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-instruct"),
            prompt=prompt,
            max_tokens=max_new_tokens,
            n=n,
            temperature=temperature,

        )
        results = [r.text.strip() for r in response.choices]
        logger.info(f'Prompt responses: {results}')
        return results