import logging
import os

import re

from .base import Detector

from ..prompts.default import DEFAULT_COH_PROMPT, DEFAULT_FLU_PROMPT, DEFAUL_REL_PROMPT, DEFAULT_CON_PRROMPT

logger = logging.getLogger(__name__)


class GEval(Detector):
    metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    @staticmethod
    def parse_output(output):
        if ':' in output:
            output = output.rsplit(':', 1)[-1]
        matched = re.search("^ ?([\d\.]+)", output)
        if matched:
            try:
                score = float(matched.group(1))
            except:
                score = 0
        else:
            if ':' in output:
                output = output.rsplit(':', 1)[-1]
                matched = re.search("^ ?([\d\.]+)", output)
                if matched:
                    try:
                        score = float(matched.group(1))
                    except:
                        score = 0
            else:
                score = 0
        return score

    @staticmethod
    def normalize_score(score):
        max_score = 5  # Maximum possible score
        normalized_score = score / max_score
        return normalized_score

    def create_prompt(self, answer, summary, metric=os.getenv("GEVAL_METRIC")):
        mapping = {
            "coherence": ("coh_detailed.txt", DEFAULT_COH_PROMPT),
            "consistency": ("con_detailed.txt", DEFAULT_CON_PRROMPT),
            "fluency": ("flu_detailed.txt", DEFAULT_FLU_PROMPT),
            "relevance": ("rel_detailed.txt", DEFAUL_REL_PROMPT)
        }
        try:
            with open(os.path.dirname(os.path.abspath(__file__)) + f'/../prompts/{mapping[metric][0]}') as readfile:
                prompt = readfile.read()
        except:
            prompt = mapping[metric][1]

        cur_prompt = prompt.replace('{{Document}}', answer).replace('{{Summary}}', summary)
        return cur_prompt

    def score(self, question, answer=None, samples=None, summary=None):
        scores = {}
        samples = []
        if not answer:
            answer = self.ask_llm(question)[0]
        answer = answer.strip()
        if not summary:
            summary_prompt = f"Create a summary with 20 maximum words from {answer}"
            summary = self.ask_llm(summary_prompt)[0].strip()
        for metric in self.metrics:
            prompt = self.create_prompt(answer, summary, metric)
            answers = self.ask_llm(prompt, n=int(os.getenv("GEVAL_SAMPLING_NUMBER", 20)))
            samples.append(answers)
            all_scores = [self.parse_output(x.strip()) for x in answers]
            score = sum(all_scores) / len(all_scores)
            scores[metric.title()] = float("{:.2f}".format(self.normalize_score(score)))
        scores['Overall'] = float("{:.2f}".format(sum([v for k, v in scores.items()])/len(scores)))
        return scores, answer, samples
