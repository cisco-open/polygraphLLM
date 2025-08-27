/*
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

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
  settings?: { key: string; value: string }[];
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
  ChatProtect = "chatProtect",
  LLM_Uncertainty = "llm_uncertainty",
  SNNE = "snne",
}
