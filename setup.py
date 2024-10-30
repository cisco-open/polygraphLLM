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


from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme_path = os.path.join(os.path.dirname(__file__), 'assets', 'README_pypi.md')
with open(readme_path, 'r') as f:
    long_description = f.read()

setup(
    name='polygraphLLM',
    version='0.1.0',
    author='Ali Payani',
    author_email='apayani@cisco.com',
    url='https://github.com/cisco-open/polygraphLLM',
    license='Apache-2.0',
    description="Hallucination detection package",
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    package_data={
        '': ['*.json', '*.png', '*.js', '*.txt']
    }
)