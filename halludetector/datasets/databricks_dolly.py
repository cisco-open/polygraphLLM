from datasets import load_dataset
from .parser import Parser


class DollyParser(Parser):
    display_name = 'Databricks Dolly'
    _id = 'databricks-dolly'

    def __init__(self):
        self.dataset = load_dataset('databricks/databricks-dolly-15k')
        self.dataset = self.dataset['train']

    def display(self):
        results = []

        for element in self.dataset:
            results.append(
                {
                    'question': element['instruction'],
                    'context': element['context'],
                    'answer': element['response'],
                    'category': element['category']
                }
            )
        return {
            'data': results,
            'columns': ['question', 'context', 'answer', 'category']
        }
