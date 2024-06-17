sh runner.sh d4j_chart_single_prompt_ 5 defects4j llama3 prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j codellama:13B prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j codellama prompts/system_msg_expbug_with_funcs.txt Chart
sh runner.sh d4j_chart_single_prompt_ 1 defects4j adrienbrault/gorilla-openfunctions-v2:Q6_K prompts/system_msg_expbug_for_gorilla.txt Chart

python compute_score.py results/d4j_chart_single_prompt_*/llama3 -l java -a -v -o combined_fl_results/chart_llama3_single_prompt.json
python compute_score.py results/d4j_chart_single_prompt_1/codellama:13B -l java -a -v -o combined_fl_results/chart_codellama13B_single_prompt.json
python compute_score.py results/d4j_chart_single_prompt_1/codellama -l java -a -v -o combined_fl_results/chart_codellama_single_prompt.json
python compute_score.py results/d4j_chart_single_prompt_1/adrienbrault/gorilla-openfunctions-v2:Q6_K -l java -a -v -o combined_fl_results/chart_gorilla_single_prompt.json

python analyze_function_calls.py results/d4j_chart_single_prompt_*/llama3 -o function_call_patterns/chart_llama3_single_prompt_multi_run
python analyze_function_calls.py results/d4j_chart_single_prompt_1/codellama:13B -o function_call_patterns/chart_codellama13b_single_prompt
python analyze_function_calls.py results/d4j_chart_single_prompt_1/codellama -o function_call_patterns/chart_codellama_single_prompt
python analyze_function_calls.py results/d4j_chart_single_prompt_1/adrienbrault/gorilla-openfunctions-v2:Q6_K -o function_call_patterns/chart_gorilla_single_prompt