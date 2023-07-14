#### e1_generate_llm_results.py
Generates LLM results for experiment 1.

#### e2_generate_model_results.py
Generates SCM model results and combines these with LLM results from below into one dataframe.

#### e2_generate_llm_results.py
Generates LLM results for experiment 2.

#### e2_preprocess_mturk_results.py
Preprocesses raw results files from MTurk experiment 2 and produces clean files (aggregated_results.csv)

#### generate_llm_similarity_ratings.py
Generate category similarity ratings for LLMs.

# Helpers

#### llms.py
Helper classes for generating LLM completions using different API vendors.

#### prompts.py
Prompt formats for experiment 1, following the S3-C1-A1-Q3-O1 prompt, and experiment 2, following the S3-C1-A1-Q1-O1-T prompt.

#### helpers.py
Misc helper functions.

#### config.py
Misc config variables.
