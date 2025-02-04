python compute_weighted_score.py results/d4j_autofl_template2_*/llama3.1 \
                                 results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble3_overlap_1

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3:70b \
                                 results/d4j_autofl_template2_*/llama3.1 \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble3_overlap_2

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3:70b \
                                 results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble3_overlap_3

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/gemma2:27b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble3_overlap_4

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3 \
                                 results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble3_overlap_5

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3.1 \
                                 results/d4j_autofl_template2_*/llama3:70b \
                                 results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble4_overlap_1

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/llama3 \
                                 results/d4j_autofl_template2_*/gemma2:27b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble4_overlap_2

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3.1:70b \
                                 results/d4j_autofl_template2_*/gemma2 \
                                 results/d4j_autofl_template2_*/gemma2:27b \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s pso -o weighted_fl_results/d4j_ensemble4_overlap_3