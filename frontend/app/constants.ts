import { Detectors } from "./types";

export const API_BASE_URL = "http://127.0.0.1:5000";

export const GERENAL_ERROR_MESSAGE = "Something went wrong!";

export const EVALUATION_PRINCIPLES_HELPER_TEXT: Record<string, string> = {
  [Detectors.Chainpoll]:
    "This evaluation indicates how many times the LLM responded 'yes' to the question of whether the provided answer contains hallucinations. Lower values indicate a lower likelihood of hallucination. The scores are bounded between 0.0 and 1.0",
  [Detectors.Refchecker]:
    "This evaluation provides a single word response from the LLM, which could be 'Entailment', 'Neutral', 'Contradiction' or 'Abstain'.",
  [Detectors.G_Eval]:
    "This evaluation reflects the correctness score assigned by the LLM based on 4 different metrics for the given question-answer pair within the provided context of references. Higher values indicate greater factuality.",
  [Detectors.SelfCheck_NGram]:
    "SelfCheckGPT NGram proposes an approach based on checking the self-consistency between an LLM response and a large number of additional responses, sampled from the same LLM using the same prompt. It computes agreement by fitting a simple unigram language model and using its probabilities on the original response. There is no score range available for this method.",
  [Detectors.SelfCheck_BertScore]:
    "SelfCheckGPT Bertscore proposes an approach based on checking the self-consistency between an LLM response and a large number of additional responses, sampled from the same LLM using the same prompt. It computes agreement using BertScore. A higher score means a higher chance of being hallucinated. The score is bounded between 0.0 and 1.0.",
  [Detectors.SelfCheck_Prompt]:
    "This evaluation includes prompting an LLM to assess information consistency in a zero-shot setup. Querying an LLM aims at assessing whether the i-th sentence is supported by the sample (as the context). A higher score means a higher chance of being hallucinated. The score is bounded between 0.0 and 1.0.",
};

export const DETECTOR_PAPERS: Record<string, string> = {
  [Detectors.Chainpoll]: "https://arxiv.org/abs/2310.18344",
  [Detectors.Refchecker]: "https://arxiv.org/abs/2405.14486",
  [Detectors.G_Eval]: "https://arxiv.org/abs/2303.16634",
  [Detectors.SelfCheck_NGram]: "https://arxiv.org/abs/2303.08896",
  [Detectors.SelfCheck_BertScore]: "https://arxiv.org/abs/2303.08896",
  [Detectors.SelfCheck_Prompt]: "https://arxiv.org/abs/2303.08896",
};
