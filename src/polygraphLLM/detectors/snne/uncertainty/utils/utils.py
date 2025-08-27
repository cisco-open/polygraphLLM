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

"""Utility functions."""
import os
import re
import hashlib
import random
import logging
import argparse
import pickle
import multiprocessing
import threading
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from ..models.huggingface_models import HuggingfaceModel, LIST_SUPPORT_MODELS


BRIEF_PROMPTS = {
    'qa': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n',
    'xsum': "Here's the text and it's short one-sentence summary.\n\n",
    'aeslc': "Write a short subject line for the email. Output only the subject line itself.\n\n",
    'de-en': "Here is a sentence in German language and its translation in English language.\n\n",
    'fr-en': "Here is a sentence in French language and its translation in English language.\n\n"
}


def set_all_seeds(seed):
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser(stages=['generate', 'compute']):
    entity = os.getenv('WANDB_SEM_UNC_ENTITY', None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument('--entity', type=str, default=entity)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="squad",
        choices=['squad', 'llm', 'llm_gpt-3.5', 'llm_gpt-4', 'squad_raw', 'rougel', 'bertscore'],
        help="Metric to assign accuracy to generations.")
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction, default=True,
        help="Compute accuracy at all temperatures or only t<<1.")
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    parser.add_argument(
        "--suffix", type=str, default='', 
        help="Additional name",)
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, default="Llama-2-7b-chat", help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens", type=int, default=50,
            help="Max number of tokens generated.",
        )
        parser.add_argument(
            "--top_p", type=float, default=1.0,
            help="Maximul total token probability.",
        )
        parser.add_argument(
            "--min_p", type=float, default=0.0,
            help="Minimum token probability. Scaled by the probability of the most likely token.",
        )
        parser.add_argument(
            "--token_limit", type=int, default=4096,
            help="Max number of tokens.")
        parser.add_argument(
            "--dataset", type=str, default="trivia_qa",
            choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'svamp', 'xsum', 'aeslc', 'de-en', 'fr-en'],
            help="Dataset to use")
        parser.add_argument(
            "--num_samples", type=int, default=400,
            help="Number of samples to use")
        parser.add_argument(
            "--num_few_shot", type=int, default=5,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_hint", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--num_generations", type=int, default=10,
            help="Number of generations to use")
        parser.add_argument(
            "--temperature", type=float, default=1.0,
            help="Temperature")
        parser.add_argument(
            "--use_mc_options", type=bool, default=True,
            help="Include MC options question?")
        parser.add_argument(
            "--get_training_set_generations", default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--use_context", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--get_training_set_generations_most_likely_only", default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Only get embedding of most likely answer for training set. "
                "This is all that's needed for p_true."))
        parser.add_argument('--compute_p_true', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_prompt", default='qa', type=str, choices=['qa', 'xsum', 'aeslc', 'chat', 'de-en', 'fr-en'])
        parser.add_argument(
            "--prompt_type", default='qa', type=str, choices=['qa', 'xsum', 'aeslc', 'de-en', 'fr-en'])
        parser.add_argument(
            "--compute_uncertainties", default=True,
            action=argparse.BooleanOptionalAction,
            help='Trigger compute_uncertainty_measures.py')
        parser.add_argument(
            "--answerable_only", default=False,
            action=argparse.BooleanOptionalAction,
            help='Exclude unanswerable questions.')
        parser.add_argument(
            "--reset_seed", default=False,
            action=argparse.BooleanOptionalAction,
            help='Reset seed for every generation.')
        # Advanced decoding
        parser.add_argument(
            "--stem_flan_type", default='', 
            choices=['cot_prompt', 'pot_prompt', 'default'], type=str,
            help='Advanced prompt format')
        parser.add_argument(
            "--cot_backup", default=False,
            action=argparse.BooleanOptionalAction,
            help='Use CoT if PoT fails.')
        parser.add_argument(
            "--prompt_format", default='alpaca', type=str,
            help='Prompt format')

    if 'compute' in stages:
        parser.add_argument('--recompute_accuracy',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--eval_wandb_runid', type=str,
                            help='wandb run id of the dataset to evaluate on')
        parser.add_argument('--train_wandb_runid', type=str, default=None,
                            help='wandb run id of the dataset from which training embeddings and p_true samples will be taken')
        parser.add_argument('--num_eval_samples', type=int, default=int(1e19))
        parser.add_argument('--compute_predictive_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_context_entails_response', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--analyze_run', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--assign_new_wandb_id', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--restore_entity_eval', type=str, default=entity)
        parser.add_argument('--restore_entity_train', type=str, default=entity)
        parser.add_argument('--condition_on_question',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--strict_entailment',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument("--entailment_model", default='deberta', type=str)
        parser.add_argument(
            "--entailment_cache_id", default=None, type=str,
            help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
        parser.add_argument('--entailment_cache_only', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_true_in_compute_stage',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--reuse_entailment_model',
                            default=False, action=argparse.BooleanOptionalAction,
                            help='Use entailment model as p_true model.')
        parser.add_argument(
            "--embedding_model", type=str, default="qwen",
            choices=['qwen', 'sfr'],
            help="Pretrained embedding model.")
        parser.add_argument(
            "--semantic_similarity", type=str, default="entailment",
            choices=['entailment', 'embedding', 'exact_match', 'metric'],
            help="Similarity to find semantic groups.")
        parser.add_argument(
            "--cluster_method", type=str, default="greedy",
            choices=['greedy', 'dfs'],
            help="Method to find semantic groups.")
        parser.add_argument(
            "--cosine_threshold", type=float, default=0.5,
            help="Cosine threshold to be considered in the same semantic group.")
        # SE parameters
        parser.add_argument('--compute_cluster_assignment_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_regular_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_semantic_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_weighted_cluster_assignment_entropy',
                            default=False, action=argparse.BooleanOptionalAction)
        # SNN parameters
        parser.add_argument('--compute_snn',
            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_wsnn',
            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--snn_temperature", type=float, default=1.0,
            help="SNN temperature")
        parser.add_argument('--self_similarity',
            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--include_neutral',
            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--snn_variant", type=str, default="full",
            choices=['full', 'only_num', 'only_denom', 'num_minus_denom'],
            help="SNN variant.")
        parser.add_argument(
            "--snn_similarity_model", type=str, default="entailment",
            choices=['entailment', 'embedding'],
            help="The model used to calculate similarity between generations.")
        parser.add_argument('--snn_wo_context',
            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--metric_model", type=str, default=None, help="Model for computing accuracies",
        )

        
    if 'finetune' in stages:
        pass
    
    return parser


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def init_model(args):
    mn = args.model_name
    
    for supported_model in LIST_SUPPORT_MODELS:
        if supported_model in mn.lower():
            model = HuggingfaceModel(
                mn, stop_sequences='default',
                max_new_tokens=args.model_max_new_tokens,
                token_limit=args.token_limit)
            break
    else:
        raise ValueError(f'Unsupported model_name `{mn}`.')
    return model


def init_model_from_name(mn, max_new_tokens=50):
    if 'llama' in mn.lower() or 'falcon' in mn or 'mistral' in mn.lower():
        model = HuggingfaceModel(
            mn, stop_sequences='default',
            max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


def get_make_prompt(args):
    if args.prompt_type == 'qa':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    elif args.prompt_type == 'xsum':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Text:\n{question}\n\n"
            if answer:
                prompt += f"Summary (one sentence):\n{answer}\n\n"
            else:
                prompt += 'Summary (one sentence):\n'
            return prompt
    elif args.prompt_type == 'aeslc':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Email:\n{question}\n\n"
            if answer:
                prompt += f"Subject line:\n{answer}\n\n"
            else:
                prompt += 'Subject line:\n'
            return prompt
    elif args.prompt_type in ['de-en', 'fr-en']:
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Original:\n{question}\n"
            if answer:
                prompt += f"Translation:\n{answer}\n\n"
            else:
                prompt += 'Translation:\n'
            return prompt
    else:
        raise ValueError

    return make_prompt


def save(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')


def get_run_name(stage_name, args, old_config=None):
    if old_config is None:
        model_name = args.model_name
        dataset = args.dataset
    else:
        model_name = old_config['model_name']
        dataset = old_config['dataset']
    metric = args.metric
    if 'checkpoint' in model_name:
        model_name_components = model_name.split('/')
        model_name = model_name_components[-2] + '_' + model_name_components[-1]
    snn_variant = args.snn_variant
    snn_temp = args.snn_temperature
    self_sim = args.self_similarity
    random_seed = args.random_seed
    host_name = os.uname()[1]
    run_name = f"{stage_name}_{model_name}_{dataset}_metric-{metric}"
    
    if 'compute_uncertainties' in args:
        if args.compute_uncertainties and (args.compute_snn or args.compute_wsnn):
            run_name = f"{run_name}_snn-{snn_variant}_snntemp{snn_temp}_selfsim{self_sim}"
            
    if 'temperature' in args:
        run_name = f"{run_name}_temp{args.temperature}"
    
    if 'num_generations' in args:
        run_name = f"{run_name}_{args.num_generations}generations"
    
    run_name = f"{run_name}{args.suffix}_seed{random_seed}_{host_name}"
    
    return run_name


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
        

def is_answerable(generation):
    return len(generation['reference']['answers']['text']) > 0


def process_question_with_flan_tag(question: str, stem_flan_type: str, dataset: str = ''):
    prefix = ''
    
    if stem_flan_type == "pot_prompt":
        prefix = " Let's write a program."
    elif stem_flan_type == "cot_prompt":
        prefix = " Let's think step by step."

    return question + prefix


def remove_flan_tag(question: str, stem_flan_type: str):
    if stem_flan_type == "pot_prompt":
        question = question.replace(" Let's write a program.", "")
    elif stem_flan_type == "cot_prompt":
        question = question.replace(" Let's think step by step.", "")
    return question


def execute_with_timeout(code: str, timeout: int=5, use_process: bool = True):
    executor = CodeExecutor(code, timeout, use_process)
    s = executor.run()
    return s


def fix_print_typo(code):
    # Define a regular expression pattern to find "print ("
    pattern = r'\bprint\s+\('
    
    # Replace "print (" with "print("
    fixed_code = re.sub(pattern, 'print(', code)
    
    return fixed_code


def truncate_after_last_print(code_str):
    # Split the code into lines
    lines = code_str.strip().splitlines()
    if len(lines) == 0:
        return ''

    # Check if the last line contains a print statement
    if lines[-1].strip().startswith("print"):
        return code_str.strip()
    
    # If not, truncate the code after the last print statement
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("print"):
            return '\n'.join(lines[:i+1])
    
    # If there's no print statement at all, return the original code
    return code_str.strip()


def find_first_relevant_split(code, split_pattern, desired_pattern):
    # Split the code based on occurrences of "```python"
    splits = code.split(split_pattern)

    # Initialize a variable to store the first split with non-whitespace characters
    first_non_whitespace_split = None

    for split in splits:
        stripped_split = split.strip()

        # Skip empty or whitespace-only splits
        if not stripped_split:
            continue

        # Store the first non-whitespace split if not already found
        if first_non_whitespace_split is None:
            first_non_whitespace_split = stripped_split

        # Check if the split contains the string 'print'
        if desired_pattern in stripped_split:
            return stripped_split

    # If no split contains 'print', return the first non-whitespace split
    return first_non_whitespace_split


def format_code(code_str: str):
    # Remove ```python
    # code_str = code_str.replace('```python', '')
    code_str = find_first_relevant_split(code_str, '```python', 'print')
    # Remove ```
    code_str = find_first_relevant_split(code_str, '```', 'print')
    # Model specific string
    # gemma-2-2b-it
    code_str = code_str.split('**Explanation')[0]
    code_str = code_str.split('**Note')[0]
    code_str = code_str.split('Let me know')[0]
    # Truncate other texts after the last print
    code_str = truncate_after_last_print(code_str)
    # Fix space in print
    # code_str = fix_print_typo(code_str)
    code = 'import math\nfrom math import *\nimport numpy as np\nimport hashlib\ndef run_it():\n'
    for line in code_str.split('\n'):
        code += '  ' + line + '\n'
    code += 'run_it()'
    return code


class CodeExecutor:
    def __init__(self, code, timeout, use_process: bool):
        self.code = format_code(code)
        print(f"### Format code: \n{self.code}")
        self.timeout = timeout
        self.error = ''
        self.use_process = use_process

    def execute_code(self, return_val):
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(self.code, globals(), locals())
            s = f.getvalue()
            s = s.strip('\n')
            return_val['result'] = s
        except Exception as e:
            print(f"Execution error:\n{e}")
            pass

    @staticmethod
    def execute_code_with_string(code, index, return_val):
        code = format_code(code)
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(code, globals(), locals())
            s = f.getvalue()
            s = s.strip('\n')
            return_val[index] = s
        except Exception as e:
            print(f"Execution error:\n{e}")
            pass

    def run(self):
        if self.use_process:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            process = multiprocessing.Process(
                target=self.execute_code, args=(return_dict,))
            process.start()
            process.join(timeout=self.timeout)
            process.terminate()
        else:
            return_dict = {}
            thread = threading.Thread(
                target=self.execute_code, args=(return_dict,))
            thread.start()
            thread.join(timeout=self.timeout)
            if thread.is_alive():
                thread.join()  # Ensures the thread is terminated before continuing
                print('time out!')
                self.error = 'Execution timed out'

        if 'result' in return_dict:
            return return_dict['result']
        else:
            return ''