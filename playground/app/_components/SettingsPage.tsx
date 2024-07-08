"use client";

import React, { useEffect, useState } from "react";
import { InfoCircledIcon } from "@radix-ui/react-icons";
import {
  TextField,
  Flex,
  Button,
  SegmentedControl,
  Text,
  Tooltip,
} from "@radix-ui/themes";
import { useMutation } from "@tanstack/react-query";
import toast, { Toaster } from "react-hot-toast";
import { updateSettings } from "../utils";
import { GERENAL_ERROR_MESSAGE } from "../constants";
import { SettingsItem } from "../types";

export const SettingsPage = ({ settings }: { settings: SettingsItem[] }) => {
  const sections = Array.from(new Set(settings.map((item) => item.section)));
  const [activeSection, setActiveSection] = useState(settings[0].section);
  const [activeSectionData, setActiveSectionData] = useState<SettingsItem[]>(
    []
  );

  useEffect(() => {
    if (activeSection) {
      setActiveSectionData(
        settings.filter((item) => item.section === activeSection)
      );
    }
  }, [activeSection]);

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

  const handleCancelClick = () => {
    setActiveSectionData(
      settings.filter((item) => item.section === activeSection)
    );
  };

  const {
    mutate: updateSettingsMutation,
    isPending,
    isError,
    isSuccess,
  } = useMutation({
    mutationFn: updateSettings,
  });

  const handleSaveClick = () => {
    const data = findChangedItems(settings, activeSectionData);
    updateSettingsMutation(data);
  };

  useEffect(() => {
    if (isError) {
      toast.error(GERENAL_ERROR_MESSAGE);
    }
  }, [isError]);

  useEffect(() => {
    if (isSuccess) {
      toast.success("Settings updated successfully!");
    }
  }, [isSuccess]);

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
              <Flex direction="column" gap="2">
                <Flex align="center" gap="2">
                  <Tooltip content={item.description}>
                    <InfoCircledIcon />
                  </Tooltip>
                  <Text as="label" weight="medium">
                    {item.name}
                  </Text>
                </Flex>
                <TextField.Root
                  id={item.key}
                  value={item.value}
                  placeholder={`${item.name}...`}
                  type={item.type}
                  onChange={(e) => handleInputChange(e, item.key)}
                  required
                />
              </Flex>
            );
          })}
          <Flex gap="4" align="center" justify="end">
            <Button
              variant="ghost"
              onClick={handleCancelClick}
              disabled={
                findChangedItems(settings, activeSectionData).length === 0
              }
            >
              Cancel
            </Button>
            <Button
              onClick={handleSaveClick}
              loading={isPending}
              disabled={
                findChangedItems(settings, activeSectionData).length === 0
              }
            >
              Save
            </Button>
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