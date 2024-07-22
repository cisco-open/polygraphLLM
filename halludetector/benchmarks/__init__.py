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

from .DropParser import DropParser
from .DatabricksDollyParser import DatabricksDollyParser
from .CovidQAParser import CovidQAParser
from .DatasetParser import DatasetParser

BENCHMARKS = [
    DropParser, 
    DatabricksDollyParser, 
    CovidQAParser
]

def get_benchmarks_display_names():
    return [{"id": benchmark.id, "display_name": benchmark.display_name} for benchmark in BENCHMARKS]

def get_benchmark(id):
    for benchmark in BENCHMARKS:
        if benchmark.id == id:
            return benchmark
