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

import React, { useEffect, useState } from "react";
import { InfoCircledIcon } from "@radix-ui/react-icons";
import {
  TextField,
  Flex,
  Button,
  SegmentedControl,
  Select,
  Text,
  Tooltip,
} from "@radix-ui/themes";
import toast, { Toaster } from "react-hot-toast";
import { SettingsItem } from "../types";

export const SettingsPage = ({ settings }: { settings: SettingsItem[] }) => {
  const [localSettings, setLocalSettings] = useState<SettingsItem[]>([]);
  const sections = Array.from(new Set(settings.map((item) => item.section)));
  const [activeSection, setActiveSection] = useState(settings[0].section);
  const [activeSectionData, setActiveSectionData] = useState<SettingsItem[]>(
    []
  );

  useEffect(() => {
    const localSettings = localStorage.getItem("polygraphLLM_settings");

    if (localSettings) {
      setLocalSettings(JSON.parse(localSettings));
    } else {
      localStorage.setItem("polygraphLLM_settings", JSON.stringify(settings));
      setLocalSettings(settings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeSection) {
      setActiveSectionData(
        localSettings.filter((item) => item.section === activeSection)
      );
    }
  }, [activeSection, localSettings]);

  const handleResetClick = () => {
    localStorage.setItem("polygraphLLM_settings", JSON.stringify(settings));
    setLocalSettings(settings);
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    key: string
  ) => {
    const settingIndex = activeSectionData?.findIndex(
      (setting) => setting.key === key
    );
    if (settingIndex !== -1) {
      const updatedSettings = [...activeSectionData];
      updatedSettings[settingIndex] = {
        ...updatedSettings[settingIndex],
        value: e.target.value,
      };
      setActiveSectionData(updatedSettings);
    }
  };

  const handleSelectChange = (value: string, key: string) => {
    const settingIndex = activeSectionData?.findIndex(
      (setting) => setting.key === key
    );
    if (settingIndex !== -1) {
      const updatedSettings = [...activeSectionData];
      updatedSettings[settingIndex] = {
        ...updatedSettings[settingIndex],
        value,
      };
      setActiveSectionData(updatedSettings);
    }
  };

  const handleCancelClick = () => {
    setActiveSectionData(
      localSettings.filter((item) => item.section === activeSection)
    );
  };

  const handleSaveClick = () => {
    const data: { field_key: string; new_value: string }[] = findChangedItems(
      localSettings,
      activeSectionData
    );

    if (data?.length > 0) {
      if (data?.some((item) => item.new_value === "")) {
        toast.error("The field cannot be empty");
        return;
      }

      const updatedSettings = localSettings.map((setting) => {
        const changedItem = data.find((item) => item.field_key === setting.key);
        if (changedItem) {
          return { ...setting, value: changedItem.new_value };
        }
        return setting;
      });

      localStorage.setItem(
        "polygraphLLM_settings",
        JSON.stringify(updatedSettings)
      );
      setLocalSettings(updatedSettings);

      toast.success("Settings updated successfully!");
    }
  };

  return (
    <Flex direction="column" gap="3" width="70%">
      <SegmentedControl.Root
        defaultValue={activeSection}
        onValueChange={(section) => setActiveSection(section)}
        style={{ width: "100%" }}
      >
        {sections.map((section) => {
          return (
            <SegmentedControl.Item key={section} value={section}>
              {section}
            </SegmentedControl.Item>
          );
        })}
      </SegmentedControl.Root>
      {activeSectionData && (
        <Flex direction="column" gap="6" width="50%" mb="8" mt="4">
          {activeSectionData?.map((item) => {
            return (
              <Flex direction="column" gap="2" key={item.key}>
                <Flex align="center" gap="2">
                  <Tooltip content={item.description}>
                    <InfoCircledIcon />
                  </Tooltip>
                  <Text as="label" weight="medium">
                    {item.name}
                  </Text>
                </Flex>
                {item.type === "select" ? (
                  <Select.Root
                    onValueChange={(val) => {
                      handleSelectChange(val, item.key);
                    }}
                    defaultValue={item.value}
                  >
                    <Select.Trigger placeholder="Select type" />
                    <Select.Content>
                      {item.values
                        ?.split(", ")
                        ?.map((item) => item.replace(/,/g, ""))
                        ?.map((item, idx) => {
                          return (
                            <Select.Item value={item} key={idx}>
                              {item}
                            </Select.Item>
                          );
                        })}
                    </Select.Content>
                  </Select.Root>
                ) : (
                  <TextField.Root
                    id={item.key}
                    value={item.value}
                    placeholder={`${item.name}...`}
                    type={item.type}
                    onChange={(e) => handleInputChange(e, item.key)}
                    step={
                      item.type === "number" && item.key.includes("THRESHOLD")
                        ? 0.1
                        : undefined
                    }
                    min={0}
                    max={
                      item.type === "number" && item.key.includes("THRESHOLD")
                        ? 1
                        : undefined
                    }
                    required
                  />
                )}
              </Flex>
            );
          })}
          <Flex align="center">
            <Button variant="soft" color="red" onClick={handleResetClick}>
              Reset All
            </Button>
            <Flex gap="4" align="center" ml="auto">
              <Button
                variant="ghost"
                onClick={handleCancelClick}
                disabled={
                  findChangedItems(localSettings, activeSectionData).length ===
                  0
                }
              >
                Cancel
              </Button>
              <Button
                onClick={handleSaveClick}
                disabled={
                  findChangedItems(localSettings, activeSectionData).length ===
                  0
                }
              >
                Save
              </Button>
            </Flex>
          </Flex>
        </Flex>
      )}
      <Toaster position="top-center" />
    </Flex>
  );
};

function findChangedItems(
  settings: SettingsItem[],
  activeSectionData: SettingsItem[]
) {
  let changedItems: any = [];

  activeSectionData.forEach((activeItem) => {
    const prevItem = settings.find(
      (prevItem) => prevItem.key === activeItem.key
    );

    if (prevItem && prevItem.value !== activeItem.value) {
      changedItems.push({
        field_key: activeItem.key,
        new_value: activeItem.value,
      });
    }
  });

  return changedItems;
}
