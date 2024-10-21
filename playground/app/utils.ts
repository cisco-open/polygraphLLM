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

import { API_BASE_URL } from "./constants";
import { HallucinationDetectionPayload, UpdateSettingsPayload } from "./types";

export const detectHallucinations = async (
  payload: HallucinationDetectionPayload
) => {
  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: "POST",
    body: JSON.stringify(payload),
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("POST request to '/detect' endpoint failed");
  }

  return response.json();
};

export const updateSettings = async (payload: UpdateSettingsPayload) => {
  const response = await fetch(`${API_BASE_URL}/settings`, {
    method: "PUT",
    body: JSON.stringify(payload),
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("PUT request to '/settings' endpoint failed");
  }

  return response.json();
};

export const askLLM = async (question: string) => {
  const response = await fetch(`${API_BASE_URL}/ask-llm`, {
    method: "POST",
    body: JSON.stringify({ question }),
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("POST request to '/ask-llm' endpoint failed");
  }

  return response.json();
};
