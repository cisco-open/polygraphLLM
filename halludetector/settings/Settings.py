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

import json
import os

class Settings:
    def __init__(self, filename):
        self.filename = filename
        self.settings = self.load_settings()

    def load_settings(self):
        with open(self.filename, 'r') as f:
            return json.load(f)

    def save_settings(self):
        with open(self.filename, 'w') as f:
            json.dump(self.settings, f, indent=2)

    def update_settings(self, field_key, new_value):
        for field in self.settings:
            if field['key'] == field_key:
                field['value'] = new_value
                os.environ[field_key] = new_value
                break
        else:
            raise KeyError(f"Field with '{field_key}' not found")
        
    def get_settings_value(self, key):
        setting = next((item for item in self.settings if item['key'] == key), None)
        if setting:
            return setting.get('value')
        else:
            return None
