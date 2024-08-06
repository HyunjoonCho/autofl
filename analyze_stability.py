import random
import argparse
import pandas as pd

from compute_score import *

def get_samples(result_dirs, sample_size):
    samples = []
    for _ in range(sample_size):
        while True:
            sample = sorted(random.sample(result_dirs, args.run_count))
            if sample in samples:
                continue
            samples.append(sample)
            break
        
    return samples

def get_result_for_a_sample(sampled_dirs, language, verbose):
    label = '_'.join([dir.split('/')[1].split('_')[-1] for dir in sampled_dirs])
    json_files, autofl_scores = compute_autofl_scores(sampled_dirs, verbose=verbose)

    if args.aux:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, language,
                                            verbose=verbose)
    else:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, language, 
                                            default_aux_score=0, verbose=verbose)
    
    method_scores = assign_rank(method_scores)

    buggy_method_ranks = get_buggy_method_ranks(method_scores, key="autofl_rank")

    summary = {"label": label, "total": len(method_scores)}
    for n in range(1, 6):
        summary[f"acc@{n}"] = calculate_acc(buggy_method_ranks, key="autofl_rank", n=n)
    
    return summary
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--run_count', '-R', type=int, default=5)
    parser.add_argument('--sample_size', '-N', type=int, default=10)
    parser.add_argument('--output', '-o', type=str, default="samples")
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--verbose', '-v', action="store_true")
    parser.add_argument('--aux', '-a', action="store_true")
    args = parser.parse_args()
    assert args.language in ["java", "python"]
    
    samples = get_samples(args.result_dirs, args.sample_size)
    
    results = []
    for sample in samples:
        results.append(get_result_for_a_sample(sample, args.language, args.verbose))
    
    df = pd.DataFrame(results)
    df.to_csv(f'{args.output}_R{args.run_count}_N{args.sample_size}.csv')