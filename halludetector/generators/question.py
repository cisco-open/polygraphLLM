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

PROMPT = """Please generate a question on the given text so that when searching on Google with the question, it's possible to get some relevant information on the topics addressed in the text. Note, you just need to give the final question without quotes in one line, and additional illustration should not be included.

For example:
Input text: The Lord of the Rings trilogy consists of The Fellowship of the Ring, The Two Towers, and The Return of the King.
Output: What are the three books in The Lord of the Rings trilogy?

Input text: %s
Output: """


logger = logging.getLogger(__name__)


class QuestionGenerator:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler

    def generate(self, paragraph):
        prompt = PROMPT % paragraph
        response = self.llm_handler.ask_llm(prompt)[0]
        logger.info(f'Question generated: {response}')
        return response
