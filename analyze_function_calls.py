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
    function_calls = {}
    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir), key=lambda fname: int(fname.split('_')[1].split('.')[0]))
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
                    if not m['function_call']['name'] in function_calls:
                        function_calls[m['function_call']['name']] = [0] * 11
                    function_calls[m['function_call']['name']][index] += 1
                    index += 1
            
    return function_calls

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    args = parser.parse_args()

    data = analyze_function_calls(args.result_dirs, args.project)
    plot_horizontal_cumulative_bar_chart(data, f'{args.output}.png')

    with open(f'{args.output}.json', "w") as f:
        json.dump(data, f, indent=4)
