import json
import os
from copy import deepcopy
from datetime import datetime
from flask import Flask, render_template, request, send_file, Response, jsonify
from concurrent.futures import ThreadPoolExecutor

from halludetector import calculate_score, init_config
from halludetector.datasets import Parser, get_benchmark, benchmarks_for_UI
# init before detector so it takes the configuration
init_config(f'{os.path.dirname(os.path.realpath(__file__))}/config.json')

from halludetector.detectors.base import Detector

detector = Detector()

app = Flask(__name__)

scorer_html_mapping = {
    'Self-Check GPT Bert Score': 'result_selfcheckgpt.html',
    'Self-Check GPT NGram': 'result_selfcheckgpt.html',
    'Self-Check GPT Prompt': 'result_selfcheckgpt.html',
    'RefChecker': 'result_refchecker.html',
    'G-Eval': 'result_refchecker.html',
    'Chain Poll': 'result_chainpoll.html',

}


def store_results(data):
    report = deepcopy(data)
    for question in report:
        for result in question['results']:
            del result['score_formatted']

    file = f'{os.path.dirname(os.path.realpath(__file__))}/results/benchmark_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    with open(file, 'w') as writefile:
        writefile.write(json.dumps(report, indent=4))
    return file


def available_methods():
    result = ''
    for key in scorer_html_mapping:
        result += f'''
       <div class="input-container">
                <input class="method-checkbox" name="method" type="checkbox" id="{key}" value="{key}"/>
                <label for="{key}">{key}</label>
       </div>
        '''
    return result


def dict_to_html(data):
    html = '<table>\n'
    for key, value in data.items():
        html += f'''
            <tr>
                <td class="score-title">{key}</td>
                <td class="score">{value}</td>
            </tr>
            '''
    html += '\n</table>'
    return html


def calculate_score_thread(method, question, answer=None, samples=None, summary=None):
    score, answer, responses = calculate_score(method, question, answer, samples, summary)
    return {
        'method': method,
        'answer': answer,
        'summary': summary,
        'samples': responses,
        'score_formatted': score if not isinstance(score, dict) else dict_to_html(score),
        'score': score,
    }


def custom_sorting_key(item):
    if item == 'General':
        return (0, item)  # Return a tuple with 0 as the first element to ensure the specific word comes first
    else:
        return (1, item)  # Return a tuple with 1 as the first element for all other words


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print('requesting evaluation')
        question = request.form['question']
        answer = request.form['answer']
        methods = request.form.getlist('method')
        if len(methods) == 1:
            score, answer, responses = calculate_score(methods[0], question, answer)
            score = score if not isinstance(score, dict) else dict_to_html(score)
            return render_template(scorer_html_mapping[methods[0]], question=question, score=score,
                                   responses=responses, answer=answer, method=methods[0])
        else:
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit tasks for each method calculation
                futures = [executor.submit(calculate_score_thread, method, question, answer) for method in
                           methods]

                # Gather results from completed threads
                data = [future.result() for future in futures]

        return render_template('dashboard.html', data=data)
    return render_template('index.html', available_methods=available_methods())


@app.route('/benchmark', methods=['GET', 'POST'])
def benchmark():
    benchmarks = benchmarks_for_UI()
    if request.method == 'POST':
        print('requesting info')
        data = Parser().parse_input(request.form)
        methods = request.form.getlist('method')
        data_results = []

        if not methods:
            return render_template(
                'benchmark_results.html', data=data_results, columns=[], file=None, benchmarks=benchmarks
            )

        for item in data:
            question = item.get('question')
            answer = item.get('answer')
            samples = item.get('samples')
            context = item.get('context')
            summary = item.get('summary')
            if context:
                question += f'\nContext: {context}'

            if not answer:
                answer = detector.ask_llm(question)
            elif not question:
                if len(methods) == 1 and methods[0] == 'G-Eval':
                    # G-Eval doesn't need a question
                    pass
                else:
                    question = detector.generate_question(answer)
            if 'Self-Check GPT Bert Score' in methods and 'Self-Check GPT NGram' in methods:
                samples = detector.ask_llm(question, n=3, temperature=0.2)

            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit tasks for each method calculation
                futures = [executor.submit(calculate_score_thread, method, question, answer, samples, summary) for method in
                           methods]

                # Gather results from completed threads
                results = [future.result() for future in futures]
            data_results.append({'question': question, 'results': results})

        columns = ['Question']
        for element in data_results[0]['results']:
            columns.append(element['method'])

        file = store_results(data_results)
        return render_template('benchmark_results.html', data=data_results, columns=columns, file=file, benchmarks=benchmarks)
    return render_template('benchmark.html', available_methods=available_methods(), benchmarks=benchmarks)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    base = 'settings'
    new_data = []
    with open('config.json', 'r') as searchfile:
        data = json.loads(searchfile.read())
    tabs = list(set([element.get('tab', 'General') for element in data]))
    tabs = sorted(tabs, key=custom_sorting_key)

    if request.method == 'POST':
        updated_keys = []
        tabs_by_name = {element['name']: element['tab'] for element in data}
        for key in request.form.keys():
            if key.startswith('settings_name_'):
                idx = key.split('_')[-1]
                name = request.form.get(f'{base}_name_{idx}')
                new_data.append({
                    "name": name,
                    "value": request.form.get(f'{base}_value_{idx}'),
                    "description": request.form.get(f'{base}_description_{idx}'),
                    "secret": True if f'{base}_secret_{idx}' in request.form.keys() else False,
                    "tab": tabs_by_name[name]
                })
                updated_keys.append(name)

        # copy what is not updated.
        for element in data:
            if element['name'] not in updated_keys:
                new_data.append(element)

        with open('config.json', 'w') as writefile:
            writefile.write(json.dumps(new_data, indent=4))
        init_config(f'{os.path.dirname(os.path.realpath(__file__))}/config.json', force=True)
        return render_template('settings.html', settings=new_data, tabs=tabs)

    return render_template('settings.html', settings=data, tabs=tabs)


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


if __name__ == '__main__':
    app.run(debug=False)
