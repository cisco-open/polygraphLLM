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

import React from "react";
import { Tabs, Box, Heading, Flex, Callout } from "@radix-ui/themes";
import { InfoCircledIcon } from "@radix-ui/react-icons";
import { SettingsPage } from "./_components/SettingsPage";
import { QuestionAnswerDetection } from "./_components/QuestionAnswerDetection";
import { BenchmarksDetection } from "./_components/BenchmarksDetection";
import styles from "./page.module.css";
import { API_BASE_URL } from "./constants";

export default async function Home() {
  const [detectorsResponse, benchmarksResponse, settingsResponse] =
    await Promise.all([
      fetch(`${API_BASE_URL}/detectors`, { cache: "no-store" }),
      fetch(`${API_BASE_URL}/benchmarks`, { cache: "no-store" }),
      fetch(`${API_BASE_URL}/settings`, { cache: "no-store" }),
    ]);
  const detectors = await detectorsResponse.json();
  const benchmarks = await benchmarksResponse.json();
  const settings = await settingsResponse.json();

  return (
    <main className={styles.main}>
      <Box>
        <Heading size="7" mb="3">
          FactualLLM
        </Heading>
        <Flex width="50%">
          <Callout.Root>
            <Callout.Icon>
              <InfoCircledIcon />
            </Callout.Icon>
            <Callout.Text size="1">
              In the context of Large Language Models (LLMs), hallucination
              refers to the generation of text that includes information or
              details that are not supported by the input or context provided to
              the model. Hallucinations occur when the model produces text that
              is incorrect, irrelevant, or not grounded in reality based on the
              input it receives.{" "}
              <strong>FactualLLM can help you detect hallucinations.</strong>
            </Callout.Text>
          </Callout.Root>
        </Flex>
      </Box>
      <Tabs.Root defaultValue="qa_detection" style={{ width: "100%" }}>
        <Tabs.List>
          <Tabs.Trigger value="qa_detection">Q&A Detection</Tabs.Trigger>
          <Tabs.Trigger value="benchmarks_detection">
            Benchmarks Detection
          </Tabs.Trigger>
          <Tabs.Trigger value="settings">Settings</Tabs.Trigger>
        </Tabs.List>
        <Box pt="6">
          <Tabs.Content value="qa_detection">
            <QuestionAnswerDetection detectors={detectors.data} />
          </Tabs.Content>
          <Tabs.Content value="benchmarks_detection">
            <BenchmarksDetection
              detectors={detectors.data}
              benchmarks={benchmarks.data}
            />
          </Tabs.Content>
          <Tabs.Content value="settings">
            <SettingsPage settings={settings.data} />
          </Tabs.Content>
        </Box>
      </Tabs.Root>
    </main>
  );
}
