import json
import os
from copy import deepcopy
from datetime import datetime
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from halludetector import calculate_score, init_config
from halludetector.datasets import get_benchmark
from halludetector.benchmarks import get_benchmark, get_benchmarks_display_names
# init before detector so it takes the configuration
init_config(f'{os.path.dirname(os.path.realpath(__file__))}/config.json')

from halludetector.detectors.base import Detector
from halludetector.detectors import get_detector
from halludetector.detectors import get_detectors_display_names
from halludetector.settings.Settings import Settings
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
                        score, answer, responses = detector.score(question, answer, samples, context)
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
    answer = detector.ask_llm(question)

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
