from typing import List

from .checker_base import CheckerBase


GPT4_CHECKING_PROMPT_Q = \
"""I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""

GPT4_CHECKING_PROMPT = \
"""I have a claim that made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""


class GPT4Checker(CheckerBase):
    def __init__(self, sentence_extractor, llm_handler) -> None:
        super().__init__(sentence_extractor)
        self.prompt_temp = GPT4_CHECKING_PROMPT
        self.prompt_temp_wq = GPT4_CHECKING_PROMPT_Q
        self.llm_handler = llm_handler

    def _check(
        self, 
        claims: List, 
        references: List,
        response: str,
        question: str = None,
    ):
        ret_labels = []
        for claim, reference in zip(claims, references):
            if isinstance(claim, list):
                assert len(claim) == 3
                claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
            if question is None:
                prompt = self.prompt_temp.format(
                    reference=reference,
                    claim=claim
                )
            else:
                prompt = self.prompt_temp_wq.format(
                    question=question,
                    reference=reference,
                    claim=claim
                )
            openai_response = self.llm_handler.ask_llm(
                prompt=prompt,
            )[0]
            if openai_response and len(openai_response):
                label = None
                if self.label_contradiction.lower() in openai_response.lower():
                    label = self.label_contradiction
                elif self.label_entailment.lower() in openai_response.lower():
                    label = self.label_entailment
                else:
                    label = self.label_neutral
                ret_labels.append(label)
            else:
                raise 'OpenAI API returns None or empty string'
        return ret_labels
