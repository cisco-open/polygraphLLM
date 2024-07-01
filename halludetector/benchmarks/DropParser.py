from datasets import load_dataset
import uuid
from .DatasetParser import DatasetParser

class DropParser(DatasetParser):
    display_name = 'Drop'
    id = 'drop'

    def download_data(self):
        self.dataset = load_dataset('EleutherAI/drop')
        self.dataset = self.dataset['train']


    def display(self, offset, limit):
        result = []
        for item in self.dataset:
            answer_spans = item["answer"]["spans"]
            answer_number = item["answer"]["number"]
            if not answer_spans and not answer_number:
                answer = item["answer"]["date"]["year"]
            else:
                if answer_spans:
                    answer = ", ".join(answer_spans)
                else:
                    answer = answer_number

            result.append({
                "id": uuid.uuid4(),
                "question": item["question"],
                "answer": answer,
                "context": item["passage"],
            })
        return self.apply_offset_limit(result, offset, limit)
