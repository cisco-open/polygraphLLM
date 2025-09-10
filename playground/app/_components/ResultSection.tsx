"use client";
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
        {Object.keys(result[0].result)
          .sort((a, b) => {
            if (a === "snne") return -1;
            if (b === "snne") return 1;
            return a.localeCompare(b);
          })
          .map((method) => {
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
          {!!result[0].result[selectedResult].reasoning ? (
            <DataList.Item align="center">
              <DataList.Label minWidth="88px">Reasoning:</DataList.Label>
              <DataList.Value>
                <Flex direction="column" gap="2">
                  {Array.isArray(result[0].result[selectedResult].reasoning) ? (
                    result[0].result[selectedResult].reasoning.map(
                      (item, idx) => {
                        if (typeof item === "object" && item !== null) {
                          return (
                            <Text key={idx}>
                              {(JSON.stringify(item), null, 2)}
                            </Text>
                          );
                        } else if (
                          typeof item === "string" &&
                          item.includes("\n")
                        ) {
                          return (
                            <div key={idx}>
                              {item.split("\n").map((line, idx) => (
                                <Text key={idx} style={{ display: "block" }}>
                                  {line}
                                </Text>
                              ))}
                            </div>
                          );
                        } else {
                          return <Text key={idx}>{item}</Text>;
                        }
                      }
                    )
                  ) : (
                    <Text>
                      {typeof result[0].result[selectedResult].reasoning ===
                      "object"
                        ? JSON.stringify(
                            result[0].result[selectedResult].reasoning,
                            null,
                            2
                          )
                        : result[0].result[selectedResult].reasoning}
                    </Text>
                  )}
                </Flex>
              </DataList.Value>
            </DataList.Item>
          ) : null}
          <DataList.Item align="center">
            <DataList.Label minWidth="88px">Score:</DataList.Label>
            <ScoreDataListValue
              val={result[0].result[selectedResult].raw_score}
            />
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
              {selectedResult === Detectors.SelfCheck_NGram ? (
                "N/A"
              ) : (
                <EvaluationDataListValue
                  isHallucinated={
                    result[0].result[selectedResult].is_hallucinated
                  }
                />
              )}
            </DataList.Value>
          </DataList.Item>
        </DataList.Root>
      )}
    </Flex>
  );
};

const EvaluationDataListValue = ({
  isHallucinated,
}: {
  isHallucinated: boolean;
}) => {
  const evaluation = isHallucinated ? "Hallucinated ❌" : "Not hallucinated ✅";

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
