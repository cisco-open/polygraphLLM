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

import logging

from .base import Detector
import re
from collections import defaultdict


CHECKED = 0
INCONSISTENT = 1
SPURIOUS = -1


logger = logging.getLogger(__name__)


class ChatProtect(Detector):
    id = 'chatProtect'
    display_name = 'ChatProtect'

    def __init__(self):
        super().__init__()


    def label_to_tag(self, label):
        if label == INCONSISTENT:
            tag = "strong"
        elif label == CHECKED:
            tag = "ok"
        else:
            tag = "ok"
        return tag
    
    def explain_consistent_cot(self, stmt1, stmt2, target, prefix):
        """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
        explain_prompt = f"""\
            I give you the beginning of a text answering the prompt "{target}".
            Then follow two statements.

            Text:
            {prefix}

            Statement 1:
            {stmt1}

            Statement 2:
            {stmt2}

            Please explain whether the statements about {target} are contradictory or factually wrong.
            Provide your explanation only.
        """
        
        res = self.ask_llm(explain_prompt)
        return res


    def check_consistent_cot(self, stmt1, stmt2, target, prefix, reason):
        """Ask the model if it finds a contradiction between the sentences. Uses Chain of Thought to improve the result"""
        if stmt1 == stmt2:
            return CHECKED
        explain_prompt = f"""\
            I gave the beginning of a text answering the prompt "{target}".
            Then followed two statements.

            Text:
            {prefix}

            Statement 1:
            {stmt1}

            Statement 2:
            {stmt2}

            I asked whether the statements about {target} are contradictory.
            The explanation is:
            {reason}

            Please conclude whether the statements are contradictory with Yes or No.
        """
        conclusions_raw = self.ask_llm(explain_prompt)
        conclusions = []
        for conclusion in conclusions_raw:
            follows_yes = re.findall(r"\byes\b", conclusion.lower())
            follows_no = re.findall(r"\bno\b", conclusion.lower())
            if follows_yes and not follows_no:
                conclusions.append(INCONSISTENT)
            elif follows_no and not follows_yes:
                conclusions.append(CHECKED)
            else:
                # spurious... is ok though
                conclusions.append(CHECKED)
        return sum(conclusions) / len(conclusions)

    def generate_statement_missing_object(self, subject, predicate, target, prefix):
        """Generates a follow up statement for the description based on a triple, resembling a cloze test."""
        topic = target.replace("Please tell me about ", "")
        if not prefix:
            prefix = "There is no preceding description."
        statement_template = """
            You are a description generator. You are given the start of an description and a question that should be answered by the next sentence. You return the next sentence for the description.
            Here is the start of a description about {}:
            {}

            Please generate the next sentence of this description.
            The generated sentence must fill the gap in this Subject;Predicate;Object triple: ({}; {}; _)
            The sentence should contain as little other information as possible.
        """
        generate_sentence = self.ask_llm(statement_template.format(topic, prefix, subject, predicate))
        return generate_sentence

    def score(self, question, answer=None, samples=None, summary=None):
        sentences = self.extract_sentences(answer)
        sent_tagged_dict = defaultdict(list)
        prefix = ""
        ok, incons = 0, 0
        summary = []
        for sentence in sentences:
            if sentence in sent_tagged_dict:
                for ext_sent in sent_tagged_dict[sentence]:
                    if ext_sent.tag == "ok":
                        ok += 1
                    else:
                        incons += 1
                continue
            triples = self.extract_triplets(sentence)
            if not triples:
                triples = [answer]

            for triple in triples:
                try:
                    alt_statement = self.generate_statement_missing_object(
                        triple[0], triple[1], question, prefix
                    )[0]
                    explanation = self.explain_consistent_cot(
                        sentence, alt_statement, question, prefix
                    )[0]
                    label = self.check_consistent_cot(
                        sentence, alt_statement, question, prefix, explanation
                    )
                    tag = self.label_to_tag(label)
                    if tag == "ok":
                        ok += 1
                    else:
                        incons += 1
                        summary.append(explanation)
                except Exception as e:
                    logging.error(f"Error processing sentence: {sentence}. Error: {str(e)}")
        if incons > 0:
            summary.insert(0, f"the count of detected hallucinations is {incons}")
        else:
            summary.insert(0, "Hallucination is not detected.")
            
        return incons, answer, summary
