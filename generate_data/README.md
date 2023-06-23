#### e2_generate_model_results.py
Generates SCM model results and combines these with LLM results from below.

#### e2_generate_llm_results.py
Generates LLM results for experiment 2 using prompt formats from e2_prompt.py

#### e2_preprocess_mturk_results.py
Preprocesses raw results files from MTurk experiment 2 and produces clean files (aggregated_results.csv)

# Helpers

#### llms.py
Helper classes for generating LLM completions using different API vendors.

#### prompts.py
Prompt formats for experiment 2, following the S3-C1-A1-Q1-O1-T prompt.

#### config.py
Misc config variables and helper functions.
