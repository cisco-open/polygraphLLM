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

DEFAULT_COH_PROMPT = '''
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Coherence:
'''

DEFAULT_CON_PROMPT = '''
You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Consistency:
'''

DEFAULT_FLU_PROMPT = '''
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Fluency (1-3):
'''

DEFAUL_REL_PROMPT = '''
You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Source Text:

{{Document}}

Summary:

{{Summary}}


Evaluation Form (scores ONLY):

- Relevance:
'''

DEFAULT_CHAINPOLL_PROMPT = '''
Does the following completion contain hallucinations?
Completion: {completion}
It was based on this question:
Question: {question}
Use chain of thought to explain the completion. Rebuild the completion using your answer and check again if the completion is right.
It is mandatory that your first word in your response is yes or no as the response of the following question.
It is mandatory to explain yourself after.
Does this completion contain hallucinations?
'''

DEFAULT_SELFCHECK_WITH_PROMPT_PROMPT = '''
"Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
'''

DEFAULT_LLM_UNCERTAINTY_VANILLA_PROMPT = """
Read the question and answer and provide your confidence in this answer. 
Note: The confidence indicates how likely you think the answer is true.
Use the following format to answer:
Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%
Only the confidence, don’t give me the explanation.
Question:{question}
Answer:{answer}
Now, please provide your confidence level.
"""


DEFAULT_LLM_UNCERTAINTY_COT_PROMPT = """
Read the question and answer, analyze step by step, and provide your confidence in this answer. 
Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
Confidence: [Your confidence level, please only include the numerical number in the range of 0-100]%
Explanation: [step-by-step analysis]
Only give me the reply according to this format, don’t give me any other words.
Question:{question}
Answer:{answer}
Now, please provide your confidence level. Let’s think it step by step.
"""


DEFAULT_LLM_UNCERTAINTY_SELF_PROBING_PROMPT = """
Question:{question}
Possible Answer:{answer}
Q: How likely is the above answer to be correct? Please first show your reasoning concisely and then answer with the following format:
Confidence: [the probability of answer {answer} to be correct, not the one you think correct, please only include the numerical number in the range of 0-100%]
Explanation: [your reasoning]
"""

DEFAULT_LLM_UNCERTAINTY_MULTI_STEP_PROMPT = """
Read the question and answer, your task is to provide your confidence in the answer. Break down the task into K steps, think step by step, give your confidence in each step in the range of 0-100%, and then derive your final confidence in this answer. 
Note: The confidence indicates how likely you think your answer is true.
Use the following format to answer:
“‘Step 1: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%
...
Step K: [Your reasoning], Confidence: [ONLY the confidence value that this step is correct]%
Overall Confidence: [Your final confidence value]%”’
"""