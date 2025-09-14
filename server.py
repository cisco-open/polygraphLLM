
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
import sys

# Add src directory to Python path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src'))

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from polygraphLLM.utils.config import init_config
from polygraphLLM.utils.benchmarks import get_benchmark, get_benchmarks_display_names
from dotenv import load_dotenv

# init before detector so it takes the configuration
load_dotenv()
init_config(f'{os.path.dirname(os.path.realpath(__file__))}/config.json')

from polygraphLLM.algorithms.base import Detector
from polygraphLLM.algorithms import get_detector, get_detectors_display_names
from polygraphLLM.utils.settings.settings import Settings
import logging

detector = Detector()

app = Flask(__name__)
CORS(app, 
     origins=["http://localhost:3000"],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


@app.route('/detect', methods=['POST', "OPTIONS"])
def detect_hallucination_route():
    """
    New threshold-based detection route that returns boolean results
    """
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        print(f"OPTIONS request received from: {request.headers.get('Origin', 'Unknown')}")
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        return response
        
    try:
        data = request.get_json()
        methods = data.get('methods')
        qas = data.get('qas')
        settings = data.get('settings')

        if not methods:
            return jsonify({'error': 'Detection method not provided'}), 400

        if not qas or not isinstance(qas, list):
            return jsonify({'error': 'Invalid or empty question-answer pairs provided'}), 400

        def get_threshold_for_method(method, detector_instance):
            """Get threshold for a detection method from settings"""
            # Map method IDs to threshold setting keys
            threshold_mapping = {
                'chainpoll': 'CHAINPOLL_THRESHOLD',
                'refchecker': 'REFCHECKER_THRESHOLD',
                'self_check_gpt_bertscore': 'SELFCHECK_BERTSCORE_THRESHOLD',
                'self_check_gpt_ngram': 'SELFCHECK_NGRAM_THRESHOLD',
                'self_check_gpt_prompt': 'SELFCHECK_PROMPT_THRESHOLD',
                'g_eval': 'GEVAL_THRESHOLD',
                'chatProtect': 'CHATPROTECT_THRESHOLD',
                'llm_uncertainty': 'LLM_UNCERTAINTY_THRESHOLD',
                'snne': 'SNNE_THRESHOLD'
            }
            
            threshold_key = threshold_mapping.get(method)
            if threshold_key and settings:
                try:
                    return float(detector_instance.find_settings_value(settings, threshold_key))
                except:
                    pass
            return 0.5  # Default threshold

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

                hallucination_results = {}

                for method in methods:
                    detector_instance = get_detector(method)
                    if detector_instance:
                        threshold = get_threshold_for_method(method, detector_instance)
                        is_hallucinated, raw_score, answer, additional_data = detector_instance.detect_hallucination(
                            question, answer, samples, context, settings, threshold
                        )
                        hallucination_results[method] = {
                            'is_hallucinated': is_hallucinated,
                            'raw_score': raw_score,
                            'threshold': threshold,
                            'reasoning': additional_data
                        }
                    else:
                        return {'error': f'Invalid detection method provided: {method}'}

                # Sort results to ensure SNNE appears first
                sorted_results = {}
                if 'snne' in hallucination_results:
                    sorted_results['snne'] = hallucination_results['snne']
                for method, result in hallucination_results.items():
                    if method != 'snne':
                        sorted_results[method] = result

                return {
                    'id': id,
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'result': sorted_results
                }
            except Exception as e:
                logging.error(f'Error processing QA: {e}')
                return {'error': f'An error occurred processing QA: {e}'}

        # Execute processing in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_question_answer, qas))

        return jsonify(results)
    except Exception as e:
        logging.error(f'Error in detect_hallucination_route: {e}')
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
    settings_manager = Settings(f'{os.path.dirname(os.path.realpath(__file__))}/config.json')
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
    question_prompt = f"{question}\nPlease answer as concise as possible, in a maximum of 2 sentences."
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
