import requests
import uuid
from .DatasetParser import DatasetParser

class CovidQAParser(DatasetParser):
    display_name = 'Covid QA'
    id = 'covid-qa'

    def download_data(self):
        url = 'https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/question-answering/COVID-QA.json'
        response = requests.get(url)
        self.dataset = response.json()

    def display(self, offset, limit):
        result = []
        for item in self.dataset["data"]:
            for qa in item["paragraphs"][0]["qas"]:
                result.append({
                    "id": uuid.uuid4(),
                    "question": qa["question"],
                    "answer": qa["answers"][0]["text"],
                    "context": item["paragraphs"][0]["context"],
                })
        return self.apply_offset_limit(result, offset, limit)
    