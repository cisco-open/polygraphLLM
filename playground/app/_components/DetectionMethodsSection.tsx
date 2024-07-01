import React from "react";
import { Box, Checkbox, CheckboxCards, Flex, Text } from "@radix-ui/themes";
import { Detector } from "../types";

interface Props {
  selectedMethods: Record<string, boolean>;
  handleCheckboxCardClick: (evt: React.MouseEvent<HTMLButtonElement>) => void;
  handleSelectAllClick: (evt: React.MouseEvent<HTMLButtonElement>) => void;
  detectors: Detector[];
}

export const DetectionMethodsSection = ({
  selectedMethods,
  handleCheckboxCardClick,
  handleSelectAllClick,
  detectors,
}: Props) => {
  return (
    <Box mt="6" maxWidth="860px">
      <Text as="label" size="2">
        <Flex gap="2">
          <Checkbox
            checked={Object.keys(selectedMethods).every(
              (methodName) => selectedMethods[methodName]
            )}
            onClick={handleSelectAllClick}
          />
          Select all
        </Flex>
      </Text>
      <CheckboxCards.Root
        mt="4"
        mr="4"
        value={Object.keys(selectedMethods).filter(
          (methodName) => selectedMethods[methodName]
        )}
      >
        {detectors.map((detector) => {
          return (
            <CheckboxCards.Item
              key={detector.id}
              value={detector.id}
              onClick={handleCheckboxCardClick}
              aria-checked={selectedMethods[detector.id]}
            >
              <Text weight="bold">{detector.display_name}</Text>
            </CheckboxCards.Item>
          );
        })}
      </CheckboxCards.Root>
    </Box>
  );
};
