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
import { Box, Heading, Flex, Callout } from "@radix-ui/themes";
import { InfoCircledIcon } from "@radix-ui/react-icons";
import { API_BASE_URL } from "./constants";
import { TabNavigation } from "./_components/TabNavigation";
import styles from "./page.module.css";

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
          PolygraphLLM
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
              <strong>PolygraphLLM can help you detect hallucinations.</strong>
            </Callout.Text>
          </Callout.Root>
        </Flex>
      </Box>
      <TabNavigation
        detectors={detectors?.data}
        benchmarks={benchmarks?.data}
        settings={settings?.data}
      />
    </main>
  );
}
