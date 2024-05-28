import json
import os

import click

from openai import OpenAI

from halludetector import calculate_score, init_config

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
