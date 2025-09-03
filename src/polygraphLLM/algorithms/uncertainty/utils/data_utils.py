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

"""Data Loading Utilities."""
import os
import json
import random
import hashlib
import datasets


def sample_dataset(ds, num_samples, seed):
    print(f"Sample {num_samples} examples from {len(ds)} examples with seed = {seed}")
    random.seed(seed)
    indices = random.sample(range(len(ds)), num_samples)
    
    return ds.select(indices)


def load_ds(dataset_name, seed, add_options=None, train_num_samples=None, val_num_samples=None):
    """Load dataset."""
    user = os.environ['USER']

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': None,
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        path = f"data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "xsum":
        dataset = datasets.load_dataset("xsum", trust_remote_code=True)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['document'],
            'answers': {'text': [x['summary']]},
            'context': None,
            'id': x['id'],
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
    
    elif dataset_name == "aeslc":
        dataset = datasets.load_dataset("aeslc", trust_remote_code=True)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['email_body'],
            'answers': {'text': [x['subject_line']]},
            'context': None,
            'id': md5hash(str(x['email_body'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
    
    elif dataset_name in "de-en":
        dataset = datasets.load_dataset("wmt14", dataset_name, trust_remote_code=True)
        # Because dataset is large, sample before formatting
        train_dataset = sample_dataset(
            dataset["train"], 
            num_samples=train_num_samples,
            seed=seed)
        validation_dataset = sample_dataset(
            dataset["test"],
            num_samples=val_num_samples,
            seed=seed)
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['translation']['de'],
            'answers': {'text': [x['translation']['en']]},
            'context': None,
            'id': md5hash(str(x['translation']['de'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
    
    elif dataset_name in "fr-en":
        dataset = datasets.load_dataset("wmt14", dataset_name, trust_remote_code=True)
        # Because dataset is large, sample before formatting
        train_dataset = sample_dataset(
            dataset["train"], 
            num_samples=train_num_samples,
            seed=seed)
        validation_dataset = sample_dataset(
            dataset["test"],
            num_samples=val_num_samples,
            seed=seed)
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['translation']['fr'],
            'answers': {'text': [x['translation']['en']]},
            'context': None,
            'id': md5hash(str(x['translation']['fr'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]
    
    else:
        raise ValueError

    return train_dataset, validation_dataset