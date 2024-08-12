import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold, StratifiedKFold
from lib.repo_interface import get_repo_interface

from compute_score import *
from optimization_strategies import *

NUM_DATAFRAME_HEADER_COLS = 7

def compute_model_scores(result_dirs, project=None):
    json_status = {}
    score_results = {}
    model_list = []
    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir))
        file_iterator = tqdm(file_iterator)
        for fname in file_iterator:
            bug_name = file2bug(fname)
            if bug_name is None:
                continue            
            if project and not bug_name.startswith(project):
                continue

            json_status[bug_name] = json_status.get(bug_name, {"OK": [], "OtherError": [], "InvalidRequestError": []}) # status -> list
            score_results[bug_name] = score_results.get(bug_name, {})   # method -> model -> score info
            status_result, score_result = json_status[bug_name], score_results[bug_name]
    
            fpath = os.path.join(result_dir, fname)
            model = result_dir.split('/')[-1]
            if model not in model_list:
                model_list.append(model)
            with open(fpath, 'r') as f:
                autofl_data = json.load(f)

            prediction = autofl_data["buggy_methods"]
            pred_status = get_prediction_status(prediction)

            status_result[pred_status] = status_result.get(pred_status, [])
            status_result[pred_status].append(fpath)

            """
            Get LLM answer
            """
            final_response = autofl_data["messages"][-1]["content"]
            pred_exprs = parse_response(final_response)

            """
            Scoring
            """

            # 1. Initialize
            ri = get_repo_interface(bug_name)

            # 2. Get mactching methods
            predicted_methods = {}
            for pred_expr in pred_exprs:
                for method in ri.get_matching_method_signatures(pred_expr):
                    predicted_methods[method] = predicted_methods.get(method, [])
                    predicted_methods[method].append(pred_expr)

            # 3. Assign scores
            # Evenly distribute the score "1" to all matching methods 
            for method in predicted_methods:
                if method not in score_result:
                    score_result[method] = {"count": 0}
                if model not in score_result[method]: 
                    score_result[method][model] = 0
                score_result[method]["count"] += 1
                score_result[method][model] += 1/len(predicted_methods)

    for bug_name in score_results:
        ri = get_repo_interface(bug_name)
        all_methods = ri.method_signatures
        buggy_methods = ri.buggy_method_signatures
        score_result = score_results[bug_name]
        num_all_runs = sum([len(json_status[bug_name][s]) for s in json_status[bug_name]])
        for method in sorted(all_methods): # lexical sort
            if method not in score_result:
                score_result[method] = {"count": 0}
            for model in model_list:
                if model not in score_result[method]: 
                    score_result[method][model] = 0
                score_result[method][model] /= num_all_runs
            score_result[method]['is_buggy'] = 1 if method in buggy_methods else 0

    return model_list, json_status, score_results

def turn_dict_into_dataframe(autofl_scores, model_list):
    flattened_data = []
    for bug, methods in autofl_scores.items():
        i = 1
        for method, data in methods.items():
            row = [i, bug, method, data['is_buggy'], data['aux_score'][0], data['aux_score'][1]]
            for model in model_list:
                row.append(data.get(model, 0.0))
            flattened_data.append(row)
            i += 1
    columns = ['i', 'bug', 'method', 'desired_score', 'aux1', 'aux2']
    columns.extend(model_list)
    df = pd.DataFrame(flattened_data, columns=columns)
    
    return df

def preprocess_results(result_dirs, project, aux, lang):
    model_list, json_files, autofl_scores = compute_model_scores(result_dirs, project)

    if aux:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, lang, verbose=True)
    else:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, lang, default_aux_score=0, verbose=True)

    return turn_dict_into_dataframe(method_scores, model_list), model_list

def apply_weight_and_evaluate(autofl_scores, model_list, weights, verbose=False):
    normalizing_factor = len(model_list) / sum(weights)
    weights = [w * normalizing_factor for w in weights]
    if verbose:
        print(f'Applying weights: {weights}')
    autofl_scores_aug = autofl_scores.copy(deep=True)
    autofl_scores_aug['weighted_sum'] = autofl_scores_aug[model_list].dot(weights)

    def create_sort_key(row):
        return (-row['weighted_sum'], [-row['aux1'], -row['aux2']], row['i'], row['method'])
    autofl_scores_aug['sort_key'] = autofl_scores_aug.apply(create_sort_key, axis=1)
    autofl_scores_aug['rank'] = autofl_scores_aug.groupby('bug')['sort_key'].rank().astype(int)

    return autofl_scores_aug[autofl_scores_aug['desired_score'] == 1].groupby('bug')['rank'].min()

def get_accuracies(rank_by_bug):
   return [len(rank_by_bug[rank_by_bug <= 1]), len(rank_by_bug[rank_by_bug <= 2]), len(rank_by_bug[rank_by_bug <= 3])]

def get_wef(rank_by_bug):
    return sum(rank_by_bug),

def create_evaluation_function(score_df, model_list):
    def evaluateVotingWeights(weight):
        return get_wef(apply_weight_and_evaluate(score_df, model_list, weight))
    return evaluateVotingWeights

def reconstruct_dict_from_dataframe(score_df):
    data = {}
    for _, row in score_df.iterrows():
        bug = row['bug']
        method = row['method']
        if bug not in data:
            data[bug] = {}
        if method not in data[bug]:
            data[bug][method] = {}
        data[bug][method]['score'] = row['weighted_sum']
        data[bug][method]['aux_score'] = (row['aux1'], row['aux2'])
    return data

def verify_acc_with_existing_pipe(weighted_scores_df):
    method_scores = reconstruct_dict_from_dataframe(weighted_scores_df)
    method_scores = assign_rank(method_scores)
    buggy_method_ranks = get_buggy_method_ranks(method_scores, key="autofl_rank")

    summary = {"total": len(method_scores)}
    for n in range(1, 11):
        summary[f"acc@{n}"] = calculate_acc(buggy_method_ranks, key="autofl_rank", n=n)
    print(json.dumps(summary, indent=4))

def cross_validation(score_df, model_list, optimizer, k=10, stratified=False):
    cv_log = f'---Running {k}-fold CV---\n'

    unique_bugs = score_df['bug'].unique()
    if stratified:
        projects = [bug_id.split('_')[0] for bug_id in unique_bugs]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_indices = list(skf.split(unique_bugs, projects))
    else: 
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_indices = list(kf.split(unique_bugs))
    
    size = len(model_list)
    added_accs = []
    for i, (train_bug_indices, validation_bug_indices) in enumerate(fold_indices):
        start = time()
        train_bugs = unique_bugs[train_bug_indices]
        validation_bugs = unique_bugs[validation_bug_indices]
        
        train_set = score_df[score_df['bug'].isin(train_bugs)]
        validation_set = score_df[score_df['bug'].isin(validation_bugs)]
        
        evaluator = create_evaluation_function(train_set, model_list)
        best, log = optimizer(evaluator, size)
        accs = get_accuracies(apply_weight_and_evaluate(validation_set, model_list, best, verbose=True))
        
        if not added_accs:
            added_accs = accs[:]
        else:
            added_accs = [added_accs[i] + accs[i] for i in range(len(added_accs))]
        cv_log += f'\nFold {i + 1:2} - Raw Best Weight: {best}\tAccuracy: {accs} out of {len(validation_bug_indices)}\tTime Taken: {time() - start}\n' + log
    cv_log += f'\n---Overall Accuracies: {added_accs}---'
    
    return cv_log

def get_correpsonding_optimizer(strategy):
    if strategy == "grid":
        return grid_search
    elif strategy == "ga":
        return ga
    elif strategy == "pso":
        return pso
    else:
        return de

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--strategy', '-s', type=str, default="de")
    parser.add_argument('--cross_validation', '-cv', action="store_true")
    parser.add_argument('--aux', '-a', action="store_true")    
    args = parser.parse_args()
    assert args.language in ["java", "python"]

    path_to_dataframe = f'{args.output}.csv'
    if os.path.isfile(path_to_dataframe):
        score_df = pd.read_csv(path_to_dataframe)
        model_list = list(score_df.columns[NUM_DATAFRAME_HEADER_COLS:])
    else:
        score_df, model_list = preprocess_results(args.result_dirs, args.project, args.aux, args.language)    
        score_df.to_csv(path_to_dataframe)

    evaluator = create_evaluation_function(score_df, model_list)
    optimizer = get_correpsonding_optimizer(args.strategy)

    if args.cross_validation:
        log = cross_validation(score_df, model_list, optimizer)
    else:
        evaluator = create_evaluation_function(score_df, model_list)
        best, log = optimizer(evaluator, len(model_list))
        accs = get_accuracies(apply_weight_and_evaluate(score_df, model_list, best, verbose=True))
        log += f'\nRaw Best Weight: {best}\tAccuracy: {accs}'
    
    output_path = f'{args.output}_{args.strategy}_CV.txt' if args.cross_validation else f'{args.output}_{args.strategy}.txt' 
    with open(output_path, 'w') as f:
        f.write(log)