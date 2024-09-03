
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

import click

from openai import OpenAI

from src import calculate_score, init_config

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def calculate_scores(question_file):
    results = []
    with open(question_file, 'r') as file:
        questions = json.loads(file.read())

    with open(f'{os.path.dirname(os.path.realpath(__file__))}/data/prompt.txt', 'r') as pf:
        prompt = pf.read()

    for question in questions:
        score, _, _ = calculate_score(client, 'chainpoll', prompt, question['input'])
        results.append((question['input'], score))
    for result in results:
        print(f"Hallucination Score: {result[1]} for question: {result[0].strip()}")


@click.command()
@click.option('--file', '-f', 'file', type=click.Path(exists=True))
@click.option('--config', '-c', 'config_file', type=click.Path(exists=True))
def main(file, config_file):
    init_config(config_file)
    calculate_scores(file)


if __name__ == "__main__":
    main()
