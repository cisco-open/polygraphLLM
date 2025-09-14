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

import React, { ChangeEvent } from "react";
import { useEffect, useRef, useState } from "react";
import { useInfiniteQuery, useMutation } from "@tanstack/react-query";
import {
  Button,
  Flex,
  Radio,
  Select,
  Skeleton,
  Table,
  Text,
} from "@radix-ui/themes";
import toast, { Toaster } from "react-hot-toast";
import { API_BASE_URL, GERENAL_ERROR_MESSAGE } from "../constants";
import { DetectionMethodsSection } from "./DetectionMethodsSection";
import {
  BenchmarkItem,
  DatasetItem,
  Detector,
  HallucinationDetectionResultItem,
  SettingsItem,
} from "../types";
import { detectHallucinations } from "../utils";
import { ResultSection } from "./ResultSection";

const DATASET_LIMIT_PER_OFFSET = 10;

interface Props {
  benchmarks: BenchmarkItem[];
  detectors: Detector[];
  settings: SettingsItem[];
}

export const BenchmarksDetection = ({
  benchmarks,
  detectors,
  settings,
}: Props) => {
  const INITIAL_STATE_SELECTED_METHODS = detectors.reduce((acc, { id }) => {
    acc[id] = true;
    return acc;
  }, {} as { [key: string]: boolean });
  const [dataset, setDataset] = useState<DatasetItem[]>([]);
  const [result, setResult] = useState<HallucinationDetectionResultItem[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("covid-qa");
  const [selectedMethods, setSelectedMethods] = useState<
    Record<string, boolean>
  >(INITIAL_STATE_SELECTED_METHODS);
  const [selectedResult, setSelectedResult] = useState("");
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);

  async function dataList(pageParam = 0) {
    const res = await fetch(
      `${API_BASE_URL}/download/${selectedDataset}?offset=${pageParam}&limit=${DATASET_LIMIT_PER_OFFSET}`
    );

    if (!res.ok) {
      throw new Error("Failed to fetch data");
    }

    const data = await res.json();
    setDataset([...dataset, ...data.data]);

    return {
      ...data,
      limit: DATASET_LIMIT_PER_OFFSET,
      page: pageParam || 1,
      offset: pageParam && pageParam > 0 ? pageParam * 10 : 0,
    };
  }

  const {
    fetchNextPage,
    isFetching,
    isFetchingNextPage,
    refetch,
    isError: isDatasetError,
  } = useInfiniteQuery({
    queryKey: ["dataList"],
    initialPageParam: 0,
    queryFn: ({ pageParam }) => dataList(pageParam),
    getNextPageParam: (lastPage) => {
      const nextPage = lastPage ? lastPage.page + 1 : 0;
      return nextPage;
    },
  });

  const {
    mutate: detectHallucinationsMutation,
    isPending,
    isError: isDetectionError,
  } = useMutation({
    mutationFn: detectHallucinations,
    onSuccess: (data) => {
      setResult(data);
      const methods = Object.keys(data[0].result);
      // Set SNNE as default if available, otherwise use first method
      const defaultMethod = methods.includes("snne") ? "snne" : methods[0];
      setSelectedResult(defaultMethod as string);
    },
  });

  const getSelectedMethods = () => {
    return Object.keys(selectedMethods)
      .filter((method) => selectedMethods[method])
      .map((method) => method);
  };

  const handleSelectAllMethodsClick = (e: any) => {
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

  const handleSubmit = () => {
    const localSettings: SettingsItem[] = localStorage.getItem(
      "polygraphLLM_settings"
    )
      ? JSON.parse(localStorage.getItem("polygraphLLM_settings") as string)
      : settings;

    const keyValueSettings = localSettings.map(({ key, value }) => ({
      key,
      value,
    }));

    detectHallucinationsMutation({
      methods: getSelectedMethods(),
      qas: dataset.filter((item) => item.id === selectedItemId),
      settings: keyValueSettings,
    });
  };

  const handleRadioButtonChange = (e: ChangeEvent<HTMLInputElement>) => {
    setSelectedItemId(e.target.value);
  };

  useEffect(() => {
    setSelectedItemId(null);
    refetch();
  }, [selectedDataset, refetch]);

  useEffect(() => {
    if (dataset.length > 0) {
      if (dataset[0]?.id) {
        setSelectedItemId(dataset[0].id);
      }
    } else {
      setSelectedItemId(null);
    }
  }, [dataset]);

  useEffect(() => {
    if (result) {
      if (resultRef.current !== null) {
        resultRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [result]);

  useEffect(() => {
    if (isDatasetError || isDetectionError) {
      toast.error(GERENAL_ERROR_MESSAGE);
    }
  }, [isDatasetError, isDetectionError]);

  const submitButtonDisabled =
    !selectedItemId ||
    !selectedDataset ||
    Object.values(selectedMethods).every((method) => !method);

  return (
    <>
      <Flex direction="column" gap="1" width="200px" mb="6">
        <Text as="label">Select dataset:</Text>
        <Select.Root
          onValueChange={(dataset) => {
            setDataset([]);
            setResult([]);
            setSelectedDataset(dataset);
          }}
          defaultValue="covid-qa"
        >
          <Select.Trigger placeholder="Select dataset" />
          <Select.Content>
            <Select.Item value="drop">Drop</Select.Item>
            <Select.Item value="covid-qa">Covid-QA</Select.Item>
            <Select.Item value="databricks-dolly">Databricks dolly</Select.Item>
          </Select.Content>
        </Select.Root>
      </Flex>
      {isFetching && !isFetchingNextPage ? (
        <Flex direction="column" gap="3">
          <Skeleton width="100%" height="50px" />
          <Skeleton width="100%" height="50px" />
          <Skeleton width="100%" height="50px" />
          <Skeleton width="100%" height="50px" />
          <Skeleton width="100%" height="50px" />
        </Flex>
      ) : dataset.length > 0 ? (
        <Table.Root
          style={{
            height: "500px",
            overflow: "auto",
            border: "1px solid #e4e4e4",
            borderRadius: "8px",
          }}
        >
          <Table.Header>
            <Table.Row style={{ textAlign: "center" }}>
              <Table.ColumnHeaderCell />
              <Table.ColumnHeaderCell>Question</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Context</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Answer</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Id</Table.ColumnHeaderCell>
            </Table.Row>
          </Table.Header>
          {dataset.map((item) => {
            return (
              <Table.Body key={item.id}>
                <Table.Row style={{ textAlign: "center" }}>
                  <Table.Cell style={{ verticalAlign: "middle" }}>
                    <Radio
                      value={item.id}
                      name="selectedItem"
                      checked={selectedItemId === item.id}
                      onChange={handleRadioButtonChange}
                    />
                  </Table.Cell>
                  <Table.RowHeaderCell
                    style={{ verticalAlign: "middle", minWidth: "250px" }}
                  >
                    {item.question}
                  </Table.RowHeaderCell>
                  <CellWithSeeMore str={item.context} />
                  <CellWithSeeMore str={item.answer} />
                  <Table.Cell style={{ verticalAlign: "middle" }}>
                    {item.id}
                  </Table.Cell>
                </Table.Row>
              </Table.Body>
            );
          })}
        </Table.Root>
      ) : null}
      {dataset.length > 1 && (
        <Flex align="center" justify="center" mt="4">
          <Button
            loading={isFetchingNextPage}
            disabled={isPending}
            onClick={() => fetchNextPage()}
          >
            Load more
          </Button>
        </Flex>
      )}
      <DetectionMethodsSection
        selectedMethods={selectedMethods}
        handleCheckboxCardClick={handleCheckboxCardClick}
        handleSelectAllClick={handleSelectAllMethodsClick}
        detectors={detectors}
      />
      <Button
        mt="8"
        mb="8"
        size="3"
        loading={isPending}
        disabled={submitButtonDisabled}
        onClick={handleSubmit}
      >
        Submit
      </Button>
      {result.length > 0 && !isPending ? (
        <ResultSection
          result={result}
          resultRef={resultRef}
          selectedResult={selectedResult}
          setSelectedResult={setSelectedResult}
          detectors={detectors}
        />
      ) : null}
      <Toaster position="top-center" />
    </>
  );
};

const CellWithSeeMore = ({ str }: { str: string }) => {
  const initial_max_length = 500;
  const [showMoreActive, setShowMoreActive] = useState(
    str?.length > initial_max_length
  );

  if (!str) {
    <Table.Cell width="50%" style={{ verticalAlign: "middle" }}>
      {"-"}
    </Table.Cell>;
  }

  return (
    <>
      <Table.Cell width="50%" style={{ verticalAlign: "middle" }}>
        {showMoreActive ? str.slice(0, initial_max_length) + "..." : str}
        {str.length > initial_max_length ? (
          <Text
            onClick={() => setShowMoreActive(!showMoreActive)}
            style={{ textDecoration: "underline", cursor: "pointer" }}
            ml="2"
          >
            {!showMoreActive ? "Show less" : "Show more"}
          </Text>
        ) : null}
      </Table.Cell>
    </>
  );
};
