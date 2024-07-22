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

from .covid_qa import CovidQAParser
from .databricks_dolly import DollyParser
from .drop import DropParser
from .summeval import SummEval
from .parser import Parser

BENCHMARKS = [
    CovidQAParser,
    DropParser,
    DollyParser,
    SummEval
]


def benchmarks_for_UI():
    return [{'id': benchmark._id, 'display_name': benchmark.display_name} for benchmark in BENCHMARKS]


def get_benchmark(_id):
    for benchmark in BENCHMARKS:
        if benchmark._id == _id:
            return benchmark
