sh runner.sh d4j_chart_single_prompt_ 5 defects4j llama3 prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_three_funcs_ 5 defects4j llama3 prompts/system_msg_expbug_with_three_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j codellama:13B prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j codellama prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j adrienbrault/gorilla-openfunctions-v2:Q6_K prompts/system_msg_expbug_for_gorilla.txt Chart
sh runner.sh d4j_autofl_template2_ 5 defects4j qwen2:72b prompts/system_msg_expbug_with_three_funcs.txt

sh runner.sh d4j_chart_template3_ 5 defects4j llama3 prompts/system_msg_expbug_tool_calling.txt Chart
python compute_score.py results/d4j_chart_template3_*/llama3 -l java -a -v -o combined_fl_results/chart_llama3_template3.json

python compute_score.py results/d4j_chart_single_prompt_*/llama3 -l java -a -v -o combined_fl_results/chart_llama3_single_prompt.json
python compute_score.py results/d4j_chart_three_funcs_*/llama3 -l java -a -v -o combined_fl_results/chart_llama3_three_funcs.json
python compute_score.py results/d4j_chart_single_prompt_1/codellama:13B -l java -a -v -o combined_fl_results/chart_codellama13B_single_prompt.json
python compute_score.py results/d4j_chart_single_prompt_1/codellama -l java -a -v -o combined_fl_results/chart_codellama_single_prompt.json
python compute_score.py results/d4j_chart_single_prompt_1/adrienbrault/gorilla-openfunctions-v2:Q6_K -l java -a -v -o combined_fl_results/chart_gorilla_single_prompt.json
python compute_score.py results/d4j_autofl_three_functions_*/llama3 -l java -a -v -o combined_fl_results/d4j_llama3_three_funcs.json
python compute_score.py results/d4j_autofl_four_functions_*/llama3 -l java -a -v -o combined_fl_results/d4j_llama3_four_funcs.json
python compute_score.py results/d4j_chart_three_funcs_*/llama3:70b -l java -a -v -o combined_fl_results/chart_llama3_70B_three_funcs.json
python compute_score.py results/d4j_chart_four_funcs_*/llama3:70b -l java -a -v -o combined_fl_results/chart_llama3_70B_four_funcs.json
python compute_score.py results/d4j_chart_three_funcs_*/mixtral -l java -a -v -o combined_fl_results/chart_mixtral_three_funcs.json
python compute_score.py results/d4j_chart_four_funcs_*/mixtral -l java -a -v -o combined_fl_results/chart_mixtral_four_funcs.json
python compute_score.py results/d4j_autofl_three_functions_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_three_funcs.json
python compute_score.py results/d4j_autofl_four_functions_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_four_funcs.json
python compute_score.py results/d4j_autofl_three_functions_1/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_three_funcs_single_run.json
python compute_score.py results/d4j_autofl_three_functions_*/codellama:70b -l java -a -v -o combined_fl_results/d4j_codellama_70B_three_funcs.json
python compute_score.py results/d4j_autofl_three_functions_*/mixtral -l java -a -v -o combined_fl_results/d4j_mixtral_three_funcs.json
python compute_score.py results/d4j_autofl_four_functions_*/mixtral -l java -a -v -o combined_fl_results/d4j_mixtral_four_funcs.json
python compute_score.py results/d4j_autofl_four_functions_*/gemma2 -l java -a -v -o combined_fl_results/d4j_gemma2_four_funcs.json
python compute_score.py results/d4j_autofl_three_functions_*/gemma2 -l java -a -v -o combined_fl_results/d4j_gemma2_three_funcs.json
python compute_score.py results/d4j_autofl2_template2_*/gemma2 -l java -a -v -o combined_fl_results/d4j_gemma2_template2_exp2.json
python compute_score.py results/d4j_autofl_template2_*/phi3:medium-128k -l java -a -v -o combined_fl_results/d4j_phi3_128k_template2.json
python compute_score.py results/d4j_autofl_template2_*/phi3:medium -l java -a -v -o combined_fl_results/d4j_phi3_template2.json
python compute_score.py results/d4j_autofl_template2_*/qwen2 -l java -a -v -o combined_fl_results/d4j_qwen2_template2.json
python compute_score.py results/d4j_autofl_template2_*/qwen2:72b -l java -a -v -o combined_fl_results/d4j_qwen2_72B_template2_R1.json
python compute_score.py results/d4j_autofl_template2_*/mistral-nemo -l java -a -v -o combined_fl_results/d4j_mistral_nemo_template2.json
python compute_score.py results/d4j_autofl_template2_*/gemma2:27b -l java -a -v -o combined_fl_results/d4j_gemma2_27B_template2_R5.json
python compute_score.py results/d4j_autofl_template2_1/codellama:34b -l java -a -v -o combined_fl_results/d4j_codellama_34B_template2_R1.json
python compute_score.py results/d4j_autofl_template2_1/llama3:8b-instruct-fp16 -l java -a -v -o combined_fl_results/d4j_llama3_8B_fp16_template2_R1.json
python compute_score.py results/d4j_autofl_template2_*/llama3:8b-instruct-fp16 -l java -a -v -o combined_fl_results/d4j_llama3_8B_fp16_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/gemma2:9b-instruct-fp16 -l java -a -v -o combined_fl_results/d4j_geamm2_9B_fp16_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/llama3.1:70b -l java -a -v -o combined_fl_results/d4j_llama3.1_70B_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/llama3.1 -l java -a -v -o combined_fl_results/d4j_llama3.1_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/mistral-nemo:12b-instruct-2407-fp16 -l java -a -v -o combined_fl_results/d4j_mistral_nemo_12B_fp16_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/mixtral:8x7b-instruct-v0.1-q6_K -l java -a -v -o combined_fl_results/d4j_mixtral_q6_K_template2_R5.json
python compute_score.py results/d4j_autofl_template2_*/llama3.1:8b-instruct-fp16 -l java -a -v -o combined_fl_results/d4j_llama3.1_fp16_template2_R5.json

python compute_score.py \
    results/d4j_autofl_stability_1/llama3 \
    results/d4j_autofl_stability_2/llama3 \
    results/d4j_autofl_stability_3/llama3 \
    results/d4j_autofl_stability_4/llama3 \
    results/d4j_autofl_stability_5/llama3 \
    -l java -a -v -o combined_fl_results/d4j_llama3_stability_1R5.json

python compute_score.py \
    results/d4j_autofl_stability_6/llama3 \
    results/d4j_autofl_stability_7/llama3 \
    results/d4j_autofl_stability_8/llama3 \
    results/d4j_autofl_stability_9/llama3 \
    results/d4j_autofl_stability_10/llama3 \
    -l java -a -v -o combined_fl_results/d4j_llama3_stability_2R5.json

python compute_score.py results/d4j_autofl_template2_1/llama3 -l java -a -v -o combined_fl_results/d4j_llama3_8B_template2_R1.json
python compute_score.py results/d4j_autofl_template2_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_tem2_test.json

python compute_weighted_score.py results/d4j_autofl_template2_*/llama3 \
                                 results/d4j_autofl_template2_*/llama3:70b \
                                 results/d4j_autofl_template2_*/gemma2 \
                                 results/d4j_autofl_template2_*/mixtral \
                                 -a -l java -s grid -o weighted_fl_results/d4j_ensemble4_R5

python compute_weighted_score.py \
    results/d4j_autofl_template2_*/llama3 results/d4j_autofl_template2_*/llama3:70b \
    results/d4j_autofl_template2_*/gemma2 results/d4j_autofl_template2_*/gemma2:27b \
    results/d4j_autofl_template2_*/mistral-nemo results/d4j_autofl_template2_*/mixtral \
    -a -l java -s de -o weighted_fl_results/d4j_ensemble6_R5

python compute_weighted_score.py \
    results/d4j_autofl_template2_*/llama3 results/d4j_autofl_template2_*/llama3:70b \
    results/d4j_autofl_template2_*/llama3.1 results/d4j_autofl_template2_*/llama3.1:70b \
    results/d4j_autofl_template2_*/gemma2 results/d4j_autofl_template2_*/gemma2:27b \
    results/d4j_autofl_template2_*/mistral-nemo results/d4j_autofl_template2_*/mixtral \
    -a -l java -s de -o weighted_fl_results/d4j_ensemble8_R5

python compute_score.py results/d4j_autofl_three_functions_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_three_funcs_chart.json -p Chart
python compute_score.py results/d4j_autofl_four_functions_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_four_funcs_chart.json -p Chart

python compute_score.py results/d4j_chart_three_funcs_1/gemma2:27b -l java -a -v -o combined_fl_results/chart_gemma2_27B_three_funcs.json
python compute_score.py results/d4j_chart_four_funcs_1/gemma2:27b -l java -a -v -o combined_fl_results/chart_gemma2_27B_four_funcs.json
python compute_score.py results/d4j_chart_three_funcs_1/gemma2 -l java -a -v -o combined_fl_results/chart_gemma2_three_funcs.json
python compute_score.py results/d4j_chart_four_funcs_1/gemma2 -l java -a -v -o combined_fl_results/chart_gemma2_four_funcs.json

python compute_score.py results/d4j_autofl_template2_1/codellama:70b results/d4j_autofl_template2_1/llama3 results/d4j_autofl_template2_1/llama3:70b results/d4j_autofl_template2_1/gemma2 results/d4j_autofl_template2_1/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble5_template2.json # llama3 8B, gemma2 9B, codellama 70B, mixtral 8x7B, llama3 70B
python compute_score.py results/d4j_autofl_template2_1/codellama:70b results/d4j_autofl_template2_1/llama3 results/d4j_autofl_template2_1/llama3:70b results/d4j_autofl_template2_1/gemma2 results/d4j_autofl_template2_1/mixtral -l java -v -o combined_fl_results/d4j_ensemble5_template2.json
python compute_score.py results/d4j_autofl_template2_1/llama3 results/d4j_autofl_template2_1/llama3:70b results/d4j_autofl_template2_1/gemma2 results/d4j_autofl_template2_1/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble4_template2.json # llama3 8B, gemma2 9B, mixtral 8x7B, llama3 70B
python compute_score.py results/d4j_autofl_template2_1/llama3:70b results/d4j_autofl_template2_1/gemma2 results/d4j_autofl_template2_1/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble3_1_template2.json # gemma2 9B, mixtral 8x7B, llama3 70B
python compute_score.py results/d4j_autofl_template2_1/llama3 results/d4j_autofl_template2_1/gemma2 results/d4j_autofl_template2_1/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble3_2_template2.json # llama3 8B, gemma2 9B, mixtral 8x7B
python compute_score.py results/d4j_autofl_template2_*/llama3 results/d4j_autofl_template2_*/llama3:70b results/d4j_autofl_template2_*/gemma2 results/d4j_autofl_template2_*/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble4_R5.json 
python compute_score.py results/d4j_autofl_template2_*/llama3 results/d4j_autofl_template2_*/llama3:70b results/d4j_autofl_template2_*/gemma2 results/d4j_autofl_template2_*/mixtral -l java -v -o combined_fl_results/d4j_ensemble4_R5_no_aux.json 
python compute_score.py results/d4j_autofl_template2_*/llama3:70b results/d4j_autofl_template2_*/gemma2 results/d4j_autofl_template2_*/mixtral -l java -a -v -o combined_fl_results/d4j_ensemble3_R5.json 

python compute_score.py results/d4j_llmtest_*/llama3 -l java -a -v -o combined_fl_results/d4j_llama3_baseline.json
python compute_score.py results/d4j_llmtest_*/llama3:70b -l java -a -v -o combined_fl_results/d4j_llama3_70B_baseline.json
python compute_score.py results/d4j_llmtest_*/mixtral -l java -a -v -o combined_fl_results/d4j_mixtral_baseline.json
python compute_score.py results/d4j_llmtest_*/gemma2 -l java -a -v -o combined_fl_results/d4j_gemma2_baseline.json

python analyze_function_calls.py results/d4j_chart_single_prompt_*/llama3 -o function_call_patterns/chart_llama3_single_prompt_multi_run
python analyze_function_calls.py results/d4j_chart_three_funcs_*/llama3 -o function_call_patterns/chart_llama3_three_funcs
python analyze_function_calls.py results/d4j_chart_single_prompt_1/codellama:13B -o function_call_patterns/chart_codellama13b_single_prompt
python analyze_function_calls.py results/d4j_chart_single_prompt_1/codellama -o function_call_patterns/chart_codellama_single_prompt
python analyze_function_calls.py results/d4j_chart_single_prompt_1/adrienbrault/gorilla-openfunctions-v2:Q6_K -o function_call_patterns/chart_gorilla_single_prompt
python analyze_function_calls.py results/d4j_chart_three_funcs_1/llama3:70b -o function_call_patterns/chart_llama3_70B_three_funcs
python analyze_function_calls.py results/d4j_chart_four_funcs_1/llama3:70b -o function_call_patterns/chart_llama3_70B_four_funcs
python analyze_function_calls.py results/d4j_chart_three_funcs_1/mixtral -o function_call_patterns/chart_mixtral_three_funcs
python analyze_function_calls.py results/d4j_chart_four_funcs_1/mixtral -o function_call_patterns/chart_mixtral_four_funcs

python analyze_function_calls.py results/d4j_autofl_three_functions*/llama3 -o function_call_patterns/d4j_llama3_three_funcs_multi_run
python analyze_function_calls.py results/d4j_autofl_four_functions*/llama3 -o function_call_patterns/d4j_llama3_four_funcs_multi_run
python analyze_function_calls.py results/d4j_autofl_three_functions*/llama3:70b -o function_call_patterns/d4j_llama3_70B_three_funcs_multi_run
python analyze_function_calls.py results/d4j_autofl_four_functions*/llama3:70b -o function_call_patterns/d4j_llama3_70B_four_funcs_multi_run
python analyze_function_calls.py results/d4j_autofl_three_functions*/codellama:70b -o function_call_patterns/d4j_codellama_70B_three_funcs_multi_run
python analyze_function_calls.py results/d4j_autofl_three_functions*/mixtral -o function_call_patterns/d4j_mixtral_three_funcs
python analyze_function_calls.py results/d4j_autofl_four_functions*/mixtral -o function_call_patterns/d4j_mixtral_four_funcs
python analyze_function_calls.py results/d4j_autofl_*/gpt-3.5-turbo-0613 -o function_call_patterns/d4j_gpt35
python analyze_function_calls.py results/d4j_autofl_*/gpt-4-0613 -o function_call_patterns/d4j_gpt4
python analyze_function_calls.py results/d4j_autofl_three_functions*/gemma2 -o function_call_patterns/d4j_gemma2_three_funcs
python analyze_function_calls.py results/d4j_autofl_four_functions*/gemma2 -o function_call_patterns/d4j_gemma2_four_funcs
python analyze_function_calls.py results/d4j_autofl2_template2_*/gemma2 -o function_call_patterns/d4j_gemma2_template2_exp2
python analyze_function_calls.py results/d4j_autofl_template2_*/phi3:medium-128k -o function_call_patterns/d4j_phi3_128k_template2
python analyze_function_calls.py results/d4j_autofl_template2_*/phi3:medium -o function_call_patterns/d4j_phi3_template2

python analyze_function_calls.py results/d4j_chart_three_funcs_1/gemma2:27b -o function_call_patterns/chart_gemma2_27B_three_funcs
python analyze_function_calls.py results/d4j_chart_three_funcs_1/gemma2:27b -o function_call_patterns/chart_gemma2_27B_three_funcs
python analyze_function_calls.py results/d4j_chart_three_funcs_1/gemma2 -o function_call_patterns/chart_gemma2_three_funcs
python analyze_function_calls.py results/d4j_chart_four_funcs_1/gemma2 -o function_call_patterns/chart_gemma2_four_funcs

python analyze_execution_time.py results/d4j_autofl_template2*/llama3 -o execution_time_statistics/d4j_llama3_8B_three_funcs
python analyze_execution_time.py results/d4j_autofl_three_functions*/codellama:70b -o execution_time_statistics/d4j_codellama_70B_three_funcs
python analyze_execution_time.py results/d4j_autofl_template2*/llama3:70b -o execution_time_statistics/d4j_llama3_70B_three_funcs
python analyze_execution_time.py results/d4j_autofl_four_functions*/mixtral -o execution_time_statistics/d4j_mixtral_8x7B_four_funcs