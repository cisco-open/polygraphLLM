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


from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


requirements = parse_requirements('requirements.txt')


setup(
    name='PolygraphLLM',
    version='0.1.0',
    author='Ali Payani',
    author_email='apayani@cisco.com',
    url='https://github.com/cisco-open/polygraphLLM',
    license='MIT',  # Choose an appropriate license
    description="Hallucination detection package",
    long_description=open('README.md').read(),
    include_package_data=True,
    packages=find_packages(),
    package_data={
        '': ['*.json'],  # Include all JSON files in the package
        'logos': ['*.png'],
        'js': ['*.js'],
        'txt': ['*.txt']
    },
    install_requires=requirements
)
