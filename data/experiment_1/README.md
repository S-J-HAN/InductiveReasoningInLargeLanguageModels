# Results files
Data used for figures or final analysis.

#### scorecard.csv
Sign test results for humans and LLMs on experiment 1 argument pairs.

Source: results/experiment_1.ipynb

# Post experiment data files
Data extracted after running our MTurk/LLM experiments.

#### llm_ratings.csv
LLM ratings for all arguments in our MTurk experiment. Models include GPT-4 (gpt-4-0314), GPT-3.5 (gpt-3.5-turbo-0613, text-danvinci-003, text-danvinci-002) and GPT-3 (text-davinci-001, davinci)

Source: generate_data/e1_generate_llm_results.py

#### aggregated_human_ratings.csv
An aggregated version of 'clean_human_ratings.csv', aggregated over every argument after filtering out unreliable MTurkers.

Source: Google Colab (Post-experiment Scripts/Preprocess Human Experiment Results/Experiment 1)

#### clean_human_ratings.csv
A cleaned version of 'raw_human_ratings.csv'.

Source: Google Colab (Post-experiment Scripts/Preprocess Human Experiment Results/Experiment 1/Aggregated CSV)

#### raw_human_ratings.csv
Raw human ratings collected from our MTurk experiment.

Source: KR

# Pre experiment data files
Data used to run our MTurk/LLM experiments.

### llm_prompts.csv
S3-C1-A1-Q3-O1 style prompts for every argument used in the MTurk experiment, for every LLM under consideration.

Source: generate_data/e1_generate_llm_results.py

