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

import evaluate
from rouge_score import tokenizers


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    list_answer = []
    
    # Filter reference
    if isinstance(answers['text'], str) or isinstance(answers['text'], int):
        list_answer = [answers['text']]
    elif isinstance(answers['text'], list):
        for text in answers['text']:
            if isinstance(text, str) or isinstance(text, int):
                list_answer.append(text)
    
    reference = {'answers': {'answer_start': answer_starts, 'text': list_answer}, 'id': example['id']}
    
    return reference


def run_eval(expression, output):
    try:
        # Safely evaluate the expression
        result = eval(expression)
        output.put(result)
    except Exception as e:
        output.put(e)
        
        
def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        print('Redo llm check.')
        predicted_answer, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        print('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def get_metric(metric):
    if metric == 'squad':
        squad_metric = evaluate.load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            score = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])['f1']
            
            return 1.0 if (score >= 50.0) else 0.0
    
    elif metric == 'squad_raw':
        squad_metric = evaluate.load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            score = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])['f1']
            
            return score / 100

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    
    # Entailment
    elif metric == 'entail':
        def metric(response, example, model, strict_entailment, *args, **kwargs):
            is_true = False
            list_reference = example['reference']['answers']['text']
            question = example['question']
            
            for reference in list_reference:
                implication_1 = model.check_implication(f'{question} {response}', f'{question} {reference}', example=example)
                implication_2 = model.check_implication(f'{question} {reference}', f'{question} {response}', example=example)
                assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
                if strict_entailment:
                    is_true = (implication_1 == 2) and (implication_2 == 2)
                else:
                    implications = [implication_1, implication_2]
                    # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
                    is_true = (0 not in implications) and ([1, 1] != implications)
                if is_true:
                    break
            
            return 1.0 if is_true else 0.0
    # Rouge-L
    elif metric == 'rougel':
        rouge = evaluate.load('rouge', keep_in_memory=True)
        tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
        
        def metric(response, example, *args, **kwargs):            
            score = rouge.compute(
                predictions=[response], 
                references=example['answers']['text'], 
                rouge_types=['rougeL'], 
                tokenizer=tokenizer)['rougeL']
            
            return score
    # BertScore
    elif metric == 'bertscore':
        bert_score = evaluate.load('bertscore')
        
        def metric(response, example, *args, **kwargs):            
            score = bert_score.compute(
                predictions=[response], 
                references=example['answers']['text'], 
                model_type='microsoft/deberta-v2-xlarge-mnli')['f1']
            
            return score[0]
    else:
        raise ValueError

    return metric