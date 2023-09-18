# Data generation

Some parts of this process were run in a colab notebook:
https://colab.research.google.com/drive/1HGTIdvPRt-5YoQzkDRY7aNA0axv9Olq9?usp=sharing


The colab code typically corresponds to the first part of this project where we generated experimental stimuli and preprocessed our human data. The code in this repository corresponds to the second part, where we generate LLM results and produce our final analysis.

This code is also available in 'setup.ipynb'. Note that differences between colab and local environments might result in different random seeding etc.


#### scrub_mturk_ids.py
Anonymise our MTurk data by replacing MTurk IDs with integer IDs. Affects raw_human_ratings.csv and clean_human_ratings.csv files in both experiments.

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
