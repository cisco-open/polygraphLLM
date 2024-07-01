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
