from datasets import load_dataset
import uuid
from .DatasetParser import DatasetParser

class DatabricksDollyParser(DatasetParser):
    display_name = 'Databricks dolly'
    id = 'databricks-dolly'

    def download_data(self):
        self.dataset = load_dataset('databricks/databricks-dolly-15k')
        self.dataset = self.dataset['train']
    

    def display(self, offset, limit):
        result = []
        for item in self.dataset:
            result.append({
                "id": uuid.uuid4(),
                "question": item["instruction"],
                "answer": item["response"],
                "context": item["context"],
            })
        return self.apply_offset_limit(result, offset, limit)
