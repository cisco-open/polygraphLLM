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

import React, { useState } from "react";
import { Tabs, Box } from "@radix-ui/themes";
import { QuestionAnswerDetection } from "./QuestionAnswerDetection";
import { BenchmarksDetection } from "./BenchmarksDetection";
import { SettingsPage } from "./SettingsPage";
import { BenchmarkItem, Detector, SettingsItem } from "../types";

interface Props {
  detectors: Detector[];
  benchmarks: BenchmarkItem[];
  settings: SettingsItem[];
}

export const TabNavigation = ({ detectors, benchmarks, settings }: Props) => {
  const [activeTab, setActiveTab] = useState("qa_detection");

  return (
    <Tabs.Root defaultValue={activeTab} style={{ width: "100%" }}>
      <Tabs.List>
        <Tabs.Trigger
          value="qa_detection"
          onClick={() => setActiveTab("qa_detection")}
        >
          Q&A Detection
        </Tabs.Trigger>
        <Tabs.Trigger
          value="benchmarks_detection"
          onClick={() => setActiveTab("benchmarks_detection")}
        >
          Benchmarks Detection
        </Tabs.Trigger>
        <Tabs.Trigger value="settings" onClick={() => setActiveTab("settings")}>
          Settings
        </Tabs.Trigger>
      </Tabs.List>
      <Box pt="6">
        <Tabs.Content value="qa_detection">
          <QuestionAnswerDetection detectors={detectors} settings={settings} />
        </Tabs.Content>
        <Tabs.Content value="benchmarks_detection">
          <BenchmarksDetection
            detectors={detectors}
            benchmarks={benchmarks}
            settings={settings}
          />
        </Tabs.Content>
        <Tabs.Content value="settings">
          <SettingsPage settings={settings} />
        </Tabs.Content>
      </Box>
    </Tabs.Root>
  );
};
