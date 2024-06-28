import json
import os

from .parser import Parser


class CovidQAParser(Parser):
    display_name = 'Covid-QA'
    _id = 'covid-qa'

    def __init__(self, file='../../datasets/Covid-QA.json'):
        self.file = file
        if self.file.startswith('..'):
            self.file = f'{os.path.dirname(os.path.realpath(__file__))}/{file}'
        with open(self.file, 'r') as parsefile:
            self.data = json.loads(parsefile.read())

    def display(self):
        results = []
        for data in self.data['data']:
            for paragraph in data['paragraphs']:
                for qas in paragraph['qas']:
                    results.append({'question': qas['question'], 'id': qas['id'], 'answer': qas['answers'][0]['text']})
        return {
            'data': results,
            'columns': ['id', 'question', 'answer']
        }
