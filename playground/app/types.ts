export interface DatasetItem {
  id: string;
  question: string;
  answer: string;
  context: string;
  samples: string;
}

export interface HallucinationDetectionPayload {
  methods: string[];
  qas: Partial<DatasetItem>[];
}

export interface HallucinationDetectionResultItem {
  id: string;
  question: string;
  answer: string;
  context: string;
  result: Record<
    string,
    {
      score: number | string | Record<string, number>;
      reasoning: string[];
    }
  >;
}

export interface SettingsItem {
  section: string;
  key: string;
  type: "text" | "number" | "password" | "select";
  name: string;
  value: string;
  description: string;
  values?: string;
}

export interface UpdateSettingsPayload {
  field_key: string;
  new_value: string | number;
}
[];

export interface BenchmarkItem {
  id: string;
  display_name: string;
}

export interface Detector {
  id: string;
  display_name: string;
}

export enum Detectors {
  Chainpoll = "chainpoll",
  Refchecker = "refchecker",
  G_Eval = "g_eval",
  SelfCheck_NGram = "self_check_gpt_ngram",
  SelfCheck_BertScore = "self_check_gpt_bertscore",
  SelfCheck_Prompt = "self_check_gpt_prompt",
  LLM_Uncertainty = "llm_uncertainty",
}
