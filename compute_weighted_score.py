import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from lib.repo_interface import get_repo_interface

from compute_score import *

PATH_TO_DATAFRAME = 'ensemble4.csv'

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
    if verbose:
        print(f'Applying weights: {weights}')
    autofl_scores_aug = autofl_scores.copy(deep=True)
    autofl_scores_aug['weighted_sum'] = autofl_scores_aug[model_list].dot(weights)

    def create_sort_key(row):
        return (-row['weighted_sum'], [-row['aux1'], -row['aux2']], row['i'], row['method'])
    autofl_scores_aug['sort_key'] = autofl_scores_aug.apply(create_sort_key, axis=1)
    autofl_scores_aug['rank'] = autofl_scores_aug.groupby('bug')['sort_key'].rank().astype(int)

    rank_by_bug = autofl_scores_aug[autofl_scores_aug['desired_score'] == 1].groupby('bug')['rank'].min()
    accuracies = [len(rank_by_bug[rank_by_bug == 1]), len(rank_by_bug[rank_by_bug <= 2]), len(rank_by_bug[rank_by_bug <= 3])]
    if verbose:
        print(f'acc@1, 2, 3: {accuracies}')

    return autofl_scores_aug, accuracies

def grid_search(score_df, model_list):
    max_weight = len(model_list) # let's assume three models, for now
    granularity = 10
    step = max_weight / granularity 
    print(step)

    weights = []
    accuracies = []
    best_weight = []
    best_accs = []
    for i in range(granularity + 1):
        for j in range(granularity + 1 - i):
            for k in range(granularity + 1 - i - j):
                l = granularity - i - j - k
                _, accs = apply_weight_and_evaluate(score_df, model_list, [i * step, j * step, k * step, l * step])
                if accs > best_accs:
                    best_accs = accs
                    best_weight = [i * step, j * step, k * step, l * step]
                weights.append([i * step, j * step, k * step, l * step])
                accuracies.append(accs)
    print(f'{best_weight} achieved accuracies of {best_accs}')
    return weights, accuracies

from sklearn.linear_model import LinearRegression

def linear_regression(score_df, model_list):
    print(score_df)
    X = score_df[model_list]
    y = score_df['desired_score']
    model = LinearRegression()
    model.fit(X, y)
    print(model.intercept_, model.coef_)
    apply_weight_and_evaluate(score_df, model_list, list(model.coef_))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def visualize_space(weights, accuracies): # only works for three models scenario
    weights = np.array(weights)
    accuracies = np.array(accuracies)

    # Function to convert 3D weights to 2D points on an equilateral triangle
    def to_2d_coords(weights):
        x = 0.5 * (2 * weights[:, 1] + weights[:, 2]) / (weights[:, 0] + weights[:, 1] + weights[:, 2])
        y = (np.sqrt(3) / 2) * weights[:, 2] / (weights[:, 0] + weights[:, 1] + weights[:, 2])
        return x, y

    # Convert the 3D weights to 2D coordinates
    x, y = to_2d_coords(weights)

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Perform Delaunay triangulation to create a mesh grid over the triangular plane
    points2D = np.vstack([x, y]).T
    tri = Delaunay(points2D)

    # Create a surface plot over the triangular plane
    ax.plot_trisurf(x, y, accuracies, triangles=tri.simplices, cmap='viridis', alpha=0.8)

    # Plot the data points
    ax.scatter(x, y, accuracies, c=accuracies, cmap='viridis', edgecolor='k', marker='o')

    # Label the axes
    ax.set_xlabel('Weight X')
    ax.set_ylabel('Weight Y')
    ax.set_zlabel('Accuracy')
    ax.set_zlim(100, 140)
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))  # Adjust the number of ticks as needed

    ax.set_title('Weight Space vs Accuracy')
    plt.savefig('grid.png')

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

import random
from deap import base, creator, tools

def create_evaluation_function(score_df, model_list):
    def evaluateVotingWeights(weight):
        _, accs = apply_weight_and_evaluate(score_df, model_list, weight)
        return accs
    return evaluateVotingWeights

def create_stats_and_logbook():
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    return stats, logbook

def ga(score_df, model_list):
    POPSIZE, NUMGEN, CXPB, MUTPB = 20, 20, 0.5, 0.3

    creator.create("WeightedFitness", base.Fitness, weights=(1.0, 0.1, 0.01))
    creator.create("Individual", list, fitness=creator.WeightedFitness) # or np.ndarray?

    toolbox = base.Toolbox()
    toolbox.register("SingleWeight", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.SingleWeight, len(model_list))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", create_evaluation_function(score_df, model_list))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POPSIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats, logbook = create_stats_and_logbook()

    best = None

    for g in range(NUMGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        best = tools.selBest(pop, 1)[0] 
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return best

import operator
import math

def pso(score_df, model_list):
    POPSIZE, NUMGEN = 10, 100

    creator.create("WeightedFitness", base.Fitness, weights=(1, 0.1, 0.01))
    creator.create("Particle", list, fitness=creator.WeightedFitness, speed=list, smin=None, smax=None, best=None)

    def generate(size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        return part

    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=len(model_list), pmin=0, pmax=1, smin=-0.5, smax=0.5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=0.3, phi2=0.5)
    toolbox.register("evaluate", create_evaluation_function(score_df, model_list))

    pop = toolbox.population(n=POPSIZE)
    stats, logbook = create_stats_and_logbook()
    best = None

    for g in range(NUMGEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    return best

def de(score_df, model_list):
    POPSIZE, NUMGEN, CXPB, DW = 20, 10, 0.9, 0.8

    creator.create("WeightedFitness", base.Fitness, weights=(1.0, 0.1, 0.01))
    creator.create("Agent", list, fitness=creator.WeightedFitness)

    toolbox = base.Toolbox()
    toolbox.register("SingleWeight", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Agent, toolbox.SingleWeight, len(model_list))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", create_evaluation_function(score_df, model_list))

    def updateAgent(agent, pop, size):
        original_agent = toolbox.clone(agent)
        a, b, c = random.sample(pop, 3)
        R = random.randint(1, size) - 1
        for i in range(size):
            if i == R or random.random() < CXPB:
                agent[i] = a[i] + DW * (b[i] - c[i])
        new_fitness = toolbox.evaluate(agent)
        if list(agent.fitness.values) > new_fitness:
            agent[:] = original_agent[:]
        else:
            agent.fitness.values = new_fitness

    toolbox.register("update", updateAgent, size=len(model_list))

    pop = toolbox.population(n=POPSIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats, logbook = create_stats_and_logbook()
    best = None

    for g in range(NUMGEN):
        for agent in pop:
            toolbox.update(agent, pop)
        pop = sorted(pop, key=lambda ind: ind.fitness.values, reverse=True)
        best = tools.selBest(pop, 1)[0] 
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return best

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--aux', '-a', action="store_true")    
    args = parser.parse_args()
    assert args.language in ["java", "python"]

    if os.path.isfile(PATH_TO_DATAFRAME):
        score_df = pd.read_csv(PATH_TO_DATAFRAME)
        model_list = ['llama3', 'llama3:70b', 'gemma2', 'mixtral']
    else:
        score_df, model_list = preprocess_results(args.result_dirs, args.project, args.aux, args.language)    
        score_df.to_csv(PATH_TO_DATAFRAME)

    # best = ga(score_df, model_list)
    # best = pso(score_df, model_list)
    best = de(score_df, model_list)
    apply_weight_and_evaluate(score_df, model_list, best, verbose=True)