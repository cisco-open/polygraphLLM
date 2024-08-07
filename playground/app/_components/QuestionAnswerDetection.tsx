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

"use client";

import React, { useEffect, useRef } from "react";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Flex,
  Text,
  TextArea,
  Blockquote,
  Button,
  Box,
} from "@radix-ui/themes";
import toast, { Toaster } from "react-hot-toast";
import { GERENAL_ERROR_MESSAGE } from "../constants";
import { DetectionMethodsSection } from "./DetectionMethodsSection";
import { askLLM, detectHallucinations } from "../utils";
import {
  Detector,
  HallucinationDetectionResultItem,
  SettingsItem,
} from "../types";
import { ResultSection } from "./ResultSection";

interface Props {
  detectors: Detector[];
  settings: SettingsItem[];
}

export const QuestionAnswerDetection = ({ detectors, settings }: Props) => {
  const INITIAL_STATE_SELECTED_METHODS = detectors.reduce((acc, { id }) => {
    acc[id] = true;
    return acc;
  }, {} as { [key: string]: boolean });
  const [selectedMethods, setSelectedMethods] = useState<
    Record<string, boolean>
  >(INITIAL_STATE_SELECTED_METHODS);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<HallucinationDetectionResultItem[]>([]);
  const [selectedResult, setSelectedResult] = useState("");
  const [emptyQuestion, setEmptyQuestion] = useState(false);
  const resultRef = useRef<HTMLDivElement | null>(null);

  const getSelectedMethods = () => {
    return Object.keys(selectedMethods)
      .filter((method) => selectedMethods[method])
      .map((method) => method);
  };

  const handleSelectAllClick = (e: any) => {
    setSelectedMethods(
      detectors.reduce((acc, { id }) => {
        acc[id] = e.target.ariaChecked !== "true";
        return acc;
      }, {} as { [key: string]: boolean })
    );
  };

  const handleCheckboxCardClick = (e: any) => {
    setSelectedMethods({
      ...selectedMethods,
      [e.currentTarget.value]: !selectedMethods[e.currentTarget.value],
    });
  };

  const {
    mutate: detectHallucinationsMutation,
    isPending,
    isError,
  } = useMutation({
    mutationFn: detectHallucinations,
    onSuccess: (data) => {
      setResult(data);
      setSelectedResult(Object.keys(data[0].result)[0]);
    },
  });

  const { mutate: askLLMMutation, isPending: isAskLLMPending } = useMutation({
    mutationFn: askLLM,
    onSuccess: (data) => {
      setAnswer(data);
    },
  });

  const handleSubmit = () => {
    const localSettings: SettingsItem[] = localStorage.getItem("settings")
      ? JSON.parse(localStorage.getItem("settings") as string)
      : settings;

    const keyValueSettings = localSettings.map(({ key, value }) => ({
      key,
      value,
    }));

    detectHallucinationsMutation({
      methods: getSelectedMethods(),
      qas: [{ question, answer }],
      settings: keyValueSettings,
    });
  };

  useEffect(() => {
    const localSettings = localStorage.getItem("settings");

    if (!localSettings) {
      localStorage.setItem("settings", JSON.stringify(settings));
    }
  }, []);

  useEffect(() => {
    if (isError) {
      toast.error(GERENAL_ERROR_MESSAGE);
    }
  }, [isError]);

  useEffect(() => {
    if (question) {
      setEmptyQuestion(false);
    }
  }, [question]);

  useEffect(() => {
    if (result) {
      if (resultRef.current !== null) {
        resultRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [result]);

  const submitButtonDisabled =
    !question ||
    !answer ||
    Object.values(selectedMethods).every((method) => !method);

  return (
    <Flex direction="column" gap="7" py="0">
      <Box>
        <Flex direction="column" gap="3" width="70%">
          <Blockquote>
            Enter a question and an answer and select at least one method. The
            answer can also be generated by LLM if you click the{" "}
            <strong>"Ask AI"</strong> button.
          </Blockquote>
          <Text mt="6" weight="bold">
            Question:
          </Text>
          <TextArea
            style={{ marginRight: "80px" }}
            variant="soft"
            placeholder="Write the question..."
            value={question}
            size="3"
            resize="vertical"
            onChange={(e) => setQuestion(e.target.value)}
          />
          {emptyQuestion && (
            <Text color="red" size="2">
              Question cannot be empty.
            </Text>
          )}
          <Text mt="4" weight="bold">
            Answer:
          </Text>
          <Flex gap="4" align="center" flexGrow="1">
            <TextArea
              style={{ width: "100%" }}
              variant="soft"
              placeholder="Write the answer or ask AI..."
              resize="vertical"
              size="3"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
            />
            <Button
              loading={isAskLLMPending}
              onClick={() => {
                if (!question) {
                  setEmptyQuestion(true);
                  return;
                }
                askLLMMutation(question);
              }}
              disabled={isAskLLMPending}
            >
              Ask AI
            </Button>
          </Flex>
        </Flex>
        <DetectionMethodsSection
          selectedMethods={selectedMethods}
          handleCheckboxCardClick={handleCheckboxCardClick}
          handleSelectAllClick={handleSelectAllClick}
          detectors={detectors}
        />
        <Button
          mt="8"
          mb="8"
          size="3"
          disabled={submitButtonDisabled}
          onClick={handleSubmit}
          loading={isPending}
        >
          Submit
        </Button>
      </Box>
      {result.length > 0 && !isPending && !isError ? (
        <ResultSection
          result={result}
          resultRef={resultRef}
          selectedResult={selectedResult}
          setSelectedResult={setSelectedResult}
          detectors={detectors}
        />
      ) : null}
      <Toaster position="top-center" />
    </Flex>
  );
};
