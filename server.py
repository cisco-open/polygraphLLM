
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

import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from src import init_config
from src.datasets import get_benchmark
from src.benchmarks import get_benchmark, get_benchmarks_display_names
from dotenv import load_dotenv

# init before detector so it takes the configuration
load_dotenv()
init_config(f'{os.path.dirname(os.path.realpath(__file__))}/config.json')

from src.detectors.base import Detector
from src.detectors import get_detector
from src.detectors import get_detectors_display_names
from src.settings.settings import Settings
import logging

detector = Detector()

app = Flask(__name__)
CORS(app)


@app.route('/detect', methods=['POST'])
def detect_route():
    try:
        data = request.get_json()
        methods = data.get('methods')
        qas = data.get('qas')
        settings = data.get('settings')

        if not methods:
            return jsonify({'error': 'Detection method not provided'}), 400

        if not qas or not isinstance(qas, list):
            return jsonify({'error': 'Invalid or empty question-answer pairs provided'}), 400

        def process_question_answer(qa):
            try:
                id = qa.get('id')
                question = qa.get('question')
                answer = qa.get('answer')
                context = qa.get('context')
                samples = qa.get('samples')
                if isinstance(answer, list):
                    if answer:
                        answer = answer[0]

                if not question:
                    return {'error': 'Question not provided'}

                hallucination_scores = {}
                
                for method in methods:
                    detector = get_detector(method)
                    if detector:
                        score, answer, responses = detector.score(question, answer, samples, context, settings)
                        hallucination_scores[method] = {'score': score, 'reasoning': responses}
                    else:
                        return {'error': f'Invalid detection method provided: {method}'}

                return {
                    'id': id,
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'result': hallucination_scores
                }
            except Exception as e:
                logging.error(f'Error processing QA: {e}')
                return jsonify({'error': f'An error occurred processing QA: {e}'}), 500

        # Execute processing in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_question_answer, qas))

        return jsonify(results)
    except Exception as e:
        logging.error(f'Error in detect_hallucinations_route: {e}')
        return jsonify({'error': 'An unexpected error occurred'}), 500
    
@app.route('/download/<benchmark_id>', methods=['GET'])
def download_benchmark_data(benchmark_id):
    try:
        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 10))

        parser = get_benchmark(benchmark_id)
        if not parser:
            return jsonify({'error': f'No parser found for benchmark ID: {benchmark_id}'}), 404

        parser_instance = parser()
        data = parser_instance.display(offset, limit)

        return jsonify({"data": data})
    except Exception as e:
        logging.error(f'Error in download_benchmark_data: {e}')
        return jsonify({'error': 'An unexpected error occurred'}), 500    


@app.route('/settings', methods=['GET', 'PUT'])
def settings():
    settings_manager = Settings('config.json')
    if request.method == 'PUT':
        payload = request.json
        try:
            for item in payload:
                field_key = item.get('field_key')
                new_value = item.get('new_value')
                settings_manager.update_settings(field_key, new_value)
                settings_manager.save_settings()
            return jsonify({'message': 'Settings updated successfully'}), 201
        except KeyError as e:
            return jsonify({'error': str(e)}), 400
    elif request.method == 'GET':
        return jsonify({"data": settings_manager.settings})


@app.route('/download', methods=['GET'])
def download():
    filename = request.args['filename']
    return send_file(filename)


@app.route('/datasets', methods=['GET'])
def datasets():
    _id = request.args['id']
    benchmark_class = get_benchmark(_id)
    if benchmark_class:
        data = benchmark_class().display()

    return jsonify(data)

@app.route('/ask-llm', methods=['POST'])
def ask_llm():
    payload = request.get_json()
    question = payload.get('question')
    question_prompt = f"{question}\nPlease answer in a maximum of 2 sentences."    
    answer = detector.ask_llm(question_prompt)

    return jsonify(answer)

@app.route('/detectors', methods=['GET'])
def get_detectors_display_name():
    detectors = get_detectors_display_names()
    return jsonify({"data": detectors})

@app.route('/benchmarks', methods=['GET'])
def get_benchmarks_display_name():
        benchmarks = get_benchmarks_display_names()
        return jsonify({"data": benchmarks})


if __name__ == '__main__':
    app.run(debug=False)
