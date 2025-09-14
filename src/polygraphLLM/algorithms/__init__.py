# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from .chainpoll import ChainPoll
from .selfcheckgpt import SelfCheckGPTBertScore, SelfCheckGPTNGram, SelfCheckGPTMQAG, SelfCheckGPTPrompt
from .refchecker import RefChecker
from .geval import GEval
from .chatProtect import ChatProtect

try:
    from .uncertainty import SNNE, LLMUncertainty
except ImportError as e:
    import logging
    logging.warning(f"Could not import SNNE/LLMUncertainty due to missing dependencies: {e}")
    SNNE = None
    LLMUncertainty = None

# Import chainpoll detector
chainpoll = ChainPoll

# Import chatProtect detector  
chatProtect = ChatProtect

# Import geval detector
geval = GEval

# Import refchecker detector
refchecker = RefChecker

# Import selfcheckgpt detectors
selfcheckgpt = {
    'SelfCheckGPTBertScore': SelfCheckGPTBertScore,
    'SelfCheckGPTNGram': SelfCheckGPTNGram,
    'SelfCheckGPTMQAG': SelfCheckGPTMQAG,
    'SelfCheckGPTPrompt': SelfCheckGPTPrompt
}

DETECTORS = [
    ChainPoll, 
    SelfCheckGPTBertScore, 
    SelfCheckGPTNGram,
    SelfCheckGPTPrompt,
    RefChecker,
    GEval,
    ChatProtect,
]

# Add SNNE and LLMUncertainty if they were successfully imported
if SNNE is not None:
    DETECTORS.insert(0, SNNE)
if LLMUncertainty is not None:
    DETECTORS.append(LLMUncertainty)

def get_detectors_display_names():
    return [{"id": detector.id, "display_name": detector.display_name} for detector in DETECTORS]

def get_detector(id):
    for detector in DETECTORS:
        if detector.id == id:
            return detector()
