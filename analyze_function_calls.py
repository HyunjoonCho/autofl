import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def file2bug(json_file):
    if not json_file.endswith(".json"):
        return None
    try:
        return os.path.basename(json_file).removeprefix('XFL-').removesuffix('.json')
    except:
        return None

def analyze_function_calls(result_dirs, project=None):
    calls_by_step = {}
    total_calls = {}
    failing_calls = {}

    for result_dir in result_dirs:
        file_iterator = sorted([f for f in os.listdir(result_dir) if f.endswith('.json')], key=lambda fname: int(fname.split('_')[1].split('.')[0]))
        print(f"Processing {result_dir}...")
        for fname in file_iterator:
            bug_name = file2bug(fname)
            if bug_name is None:
                continue            
            if project and not bug_name.startswith(project):
                continue

            fpath = os.path.join(result_dir, fname)
            with open(fpath, 'r') as f:
                autofl_data = json.load(f)

            valid_messages = autofl_data["messages"]
            
            index = 0
            for m in valid_messages:
                if m['role'] == 'assistant' and 'function_call' in m and index < 11:
                    if not m['function_call']['name'] in calls_by_step:
                        calls_by_step[m['function_call']['name']] = [0] * 11
                    calls_by_step[m['function_call']['name']][index] += 1
                    index += 1

            for i in range(len(valid_messages) - 1):
                m = valid_messages[i]
                next_m = valid_messages[i + 1]
                if m['role'] == 'assistant' and 'function_call' in m:
                    if next_m['role'] != 'function':
                        continue 
                    if not m['function_call']['name'] in total_calls:
                        total_calls[m['function_call']['name']] = 0
                        failing_calls[m['function_call']['name']] = 0
                    if 'error_message' in next_m['content']:
                        failing_calls[m['function_call']['name']] += 1
                    total_calls[m['function_call']['name']] += 1
    
    return calls_by_step, total_calls, failing_calls

def plot_horizontal_cumulative_bar_chart(data, path):
    MAX_STEPS=11
    plt.style.use('style/style-formal.mplstyle')
    func_indices = {'get_failing_tests_covered_classes': 0, 'get_failing_tests_covered_methods_for_class': 1,
          'get_code_snippet': 2, 'get_comments': 3} # for defects4j
    colors=['039dff', 'ABDEFF', 'd62728', 'EB9394', '000000']
    func_labels = ['class_cov', 'method_cov', 'snippet', 'comments', 'undefined']

    labels = list(data.keys())
    total_runs = data[labels[0]][0]
   
    undefined_functions = [l for l in labels if l not in func_indices]
    labels = sorted([l for l in labels if l in func_indices], key=lambda label: func_indices[label])
    values = [data[label] for label in labels]

    if undefined_functions:
        undefined_calls = [0] * MAX_STEPS
        for func in undefined_functions:
            to_add = data[func]
            undefined_calls = [undefined_calls[i] + to_add[i] for i in range(MAX_STEPS)]
        values.append(undefined_calls)
        labels.append('undefined_functions')

    values = np.array(values) / total_runs
    cumulative_values = np.cumsum(values, axis=0)
    y = np.arange(len(data[labels[0]]))
    
    _, ax = plt.subplots(figsize=(5, 2))
    
    for i in range(len(labels)):
        try:
            label_index = func_indices[labels[i]]
        except:
            label_index = 4
        if i == 0:
            ax.barh(y, values[i], label=func_labels[label_index], color=f'#{colors[label_index]}', tick_label=[f'Step {i}' for i in range(MAX_STEPS)])
        else:
            ax.barh(y, values[i], left=cumulative_values[i-1], label=func_labels[label_index], color=f'#{colors[label_index]}', tick_label=[f'Step {i}' for i in range(MAX_STEPS)])
    
    ax.set_xlabel('Proportion of Runs')
    ax.set_title('Function Call Distribution at Each Step')
    ax.legend()
    
    ax.set_ylim(len(data[labels[0]]) - 0.5, -0.5)
    plt.savefig(path, bbox_inches='tight')

def plot_function_calls(total_calls, failing_calls, path):
    func_indices = {'get_failing_tests_covered_classes': 0, 'get_failing_tests_covered_methods_for_class': 1,
          'get_code_snippet': 2, 'get_comments': 3} # for defects4j
    colors=['039dff', 'ABDEFF', 'd62728', 'EB9394', '000000']
    func_labels = ['class_cov', 'method_cov', 'snippet', 'comments', 'undefined']

    functions = [f for f in list(total_calls.keys()) if f in func_indices]
    total_values = [total_calls[func] for func in functions]
    failing_values = [failing_calls[func] for func in functions]

    translated_colors = [f'#{colors[func_indices[f]]}' for f in functions]
    translated_labels = [func_labels[func_indices[f]] for f in functions]

    x = np.arange(len(functions))

    plt.figure(figsize=(6, 8))
    plt.bar(x, total_values, width=0.8, label='Total Calls', color=translated_colors, alpha=0.5)
    plt.bar(x, failing_values, width=0.8, label='Failing Calls', color=translated_colors, alpha=1.0)

    plt.ylabel('Number of Calls', fontsize=14)
    plt.title('Total Calls vs Failing Calls per Function', fontsize=16)
    plt.xticks(x, translated_labels)

    plt.savefig(path, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    args = parser.parse_args()

    calls_by_step, total, failing = analyze_function_calls(args.result_dirs, args.project)
    plot_horizontal_cumulative_bar_chart(calls_by_step, f'{args.output}_distribution.png')
    plot_function_calls(total, failing, f'{args.output}_failing_rate.png')

    with open(f'{args.output}.json', "w") as f:
        json.dump({'total': total, 'failing': failing, 'steps': calls_by_step}, f, indent=4)
