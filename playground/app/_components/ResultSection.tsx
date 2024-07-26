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

import { ExternalLinkIcon, InfoCircledIcon } from "@radix-ui/react-icons";
import {
  Box,
  Callout,
  DataList,
  Flex,
  SegmentedControl,
  Text,
} from "@radix-ui/themes";
import {
  DETECTOR_PAPERS,
  EVALUATION_PRINCIPLES_HELPER_TEXT,
} from "../constants";
import Link from "next/link";
import {
  Detector,
  Detectors,
  HallucinationDetectionResultItem,
} from "../types";

interface Props {
  result: HallucinationDetectionResultItem[];
  resultRef: React.MutableRefObject<HTMLDivElement | null>;
  detectors: Detector[];
  selectedResult: string;
  setSelectedResult: (result: string) => void;
}

export const ResultSection = ({
  result,
  resultRef,
  detectors,
  selectedResult,
  setSelectedResult,
}: Props) => {
  return (
    <Flex
      direction="column"
      gap="6"
      style={{ paddingBottom: "120px" }}
      ref={resultRef}
    >
      <Text weight="bold" size="5">
        Results are ready! 🎉
      </Text>
      <SegmentedControl.Root
        defaultValue={selectedResult}
        onValueChange={(method) => setSelectedResult(method)}
      >
        {Object.keys(result[0].result).map((method) => {
          const methodData = detectors.find((item) => item.id === method);
          return (
            <SegmentedControl.Item
              key={methodData?.id}
              value={methodData?.id ?? ""}
            >
              {methodData?.display_name}
            </SegmentedControl.Item>
          );
        })}
      </SegmentedControl.Root>
      {selectedResult && (
        <Callout.Root>
          <Flex align="center" gap="3">
            <Callout.Icon>
              <InfoCircledIcon />
            </Callout.Icon>
            <Callout.Text size="1">
              {EVALUATION_PRINCIPLES_HELPER_TEXT[selectedResult]}
            </Callout.Text>
          </Flex>
        </Callout.Root>
      )}
      {selectedResult && (
        <DataList.Root>
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Question:</DataList.Label>
            <DataList.Value>{result[0].question}</DataList.Value>
          </DataList.Item>
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Answer:</DataList.Label>
            <DataList.Value>{result[0].answer}</DataList.Value>
          </DataList.Item>
          {result[0].context && (
            <DataList.Item align="center">
              <DataList.Label minWidth="88px">Context:</DataList.Label>
              <DataList.Value>{result[0].context}</DataList.Value>
            </DataList.Item>
          )}
          {!!result[0].result[selectedResult].reasoning &&
          selectedResult !== Detectors.G_Eval ? (
            <DataList.Item align="center">
              <DataList.Label minWidth="88px">Reasoning:</DataList.Label>
              <DataList.Value>
                <Flex direction="column" gap="0">
                  {result[0].result[selectedResult].reasoning.map(
                    (item, idx) => {
                      if (item.includes("\n")) {
                        return item
                          .split("\n")
                          .map((line, idx) => <Text key={idx}>{line}</Text>);
                      } else {
                        return <Text key={idx}>{item}</Text>;
                      }
                    }
                  )}
                </Flex>
              </DataList.Value>
            </DataList.Item>
          ) : null}
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Score:</DataList.Label>
            <ScoreDataListValue val={result[0].result[selectedResult].score} />
          </DataList.Item>
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Citation:</DataList.Label>
            <DataList.Value>
              <Link href={DETECTOR_PAPERS[selectedResult]} target="_blank">
                <Flex align="center" gap="1">
                  <ExternalLinkIcon />
                  Paper
                </Flex>
              </Link>
            </DataList.Value>
          </DataList.Item>
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Evaluation:</DataList.Label>
            <DataList.Value>
              <EvaluationDataListValue
                method={selectedResult as Detectors}
                score={result[0].result[selectedResult].score}
              />
            </DataList.Value>
          </DataList.Item>
        </DataList.Root>
      )}
    </Flex>
  );
};

const EvaluationDataListValue = ({
  method,
  score,
}: {
  method: Detectors;
  score: number | string | Record<string, number>;
}) => {
  let evaluation = getEvaluationLabel(method, score);

  if (method !== Detectors.SelfCheck_NGram) {
    if (evaluation.toLowerCase().includes("not")) {
      evaluation += " ✅";
    } else {
      evaluation += " ❌";
    }
  }

  return (
    <DataList.Value style={{ display: "flex", alignItems: "center" }}>
      {evaluation}
    </DataList.Value>
  );
};

const ScoreDataListValue = ({
  val,
}: {
  val: number | string | Record<string, number>;
}) => {
  if (typeof val === "object" && val !== null) {
    return (
      <DataList.Value>
        <Flex direction="column">
          {Object.entries(val).map(([key, value]) => (
            <Box key={key}>
              <strong>{key}:</strong> {value}
            </Box>
          ))}
        </Flex>
      </DataList.Value>
    );
  }

  return <DataList.Value>{val}</DataList.Value>;
};

const getEvaluationLabel = (
  method: Detectors,
  score: string | number | Record<string, number>
): string => {
  if (method === Detectors.G_Eval && typeof score === "object") {
    score = score["Total"];

    if (score === 0) {
      return "Hallucinated";
    } else if (score <= 0.5) {
      return "Most likely hallucinated";
    } else if (score === 1) {
      return "Not hallucinated";
    } else {
      return "Most likely not hallucinated";
    }
  }

  if (method === Detectors.ChatProtect) {
    if (score === 0) {
      return "Not hallucinated";
    } else {
      return "Hallucinated";
    }
  }

  if (
    (method === Detectors.Chainpoll ||
      method === Detectors.SelfCheck_BertScore ||
      method === Detectors.SelfCheck_Prompt) &&
    typeof score === "number"
  ) {
    if (score === 0) {
      return "Not hallucinated";
    } else if (score <= 0.5) {
      return "Most likely not hallucinated";
    } else if (score === 1) {
      return "Hallucinated";
    } else {
      return "Most likely hallucinated";
    }
  }

  if (method === Detectors.LLM_Uncertainty && typeof score === "string") {
    score = Number(score?.split("/")[0]);
    if (score === 100) {
      return "Not hallucinated";
    } else if (score > 50) {
      return "Most likely not hallucinated";
    } else {
      return "Most likely hallucinated";
    }
  }

  if (method === Detectors.Refchecker && typeof score === "object") {
    if (score["Entailment"] > 0.5) {
      return "Not hallucinated";
    } else if (score["Contradiction"] > 0.2) {
      return "Hallucinated";
    } else {
      return "Most likely hallucinated";
    }
  }

  return "N/A";
};
