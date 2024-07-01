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
      fetch(`${API_BASE_URL}/detectors`),
      fetch(`${API_BASE_URL}/benchmarks`),
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
