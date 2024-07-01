import requests

from .parser import Parser


class SummEval(Parser):
    display_name = 'SummEval'
    _id = 'summeval'
    url = 'https://raw.githubusercontent.com/nlpyang/geval/main/data/summeval.json'

    def __init__(self):
        response = requests.get(self.url)
        self.dataset = response.json()

    def display(self):
        results = []

        for element in self.dataset:
            results.append(
                {
                    'answer': element['source'],
                    'summary': element['system_output']
                }
            )
        return {
            'data': results,
            'columns': ['answer', 'summary']
        }
