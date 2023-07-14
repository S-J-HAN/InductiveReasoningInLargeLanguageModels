# Results files
Data used for figures or final analysis.

#### results/blockwise_participant_bootstrap.csv
A block based bootstrap of MTurker ratings vs model ratings across 1000 iterations.

Source: results/experiment_2.ipynb

#### results/blockwise_participant_bootstrap_summary.csv
Mean and SEs across all iterations in the above file.

Source: results/experiment_2.ipynb

#### results/blockwise_split_half_reliabilities.csv
A block based split-half reliability of MTurker ratings for each argument across 1000 iterations.

Source: results/experiment_2.ipynb

#### results/blockwise_split_half_reliabilities_summary.csv
Mean and SEs across all iterations in the above file.

Source: results/experiment_2.ipynb

# Post experiment data files
Data extracted after running our MTurk/LLM experiments.

#### model_ratings.csv
Model ratings for all arguments from GPT-4, GPT-3.5, GPT-3, SCM, MaxSim and MeanSim models.

Source: generate_data/e2_generate_model_results.py

#### llm_ratings.csv
LLM ratings for all arguments in our MTurk experiment. Models include GPT-4 (gpt-4-0314), GPT-3.5 (gpt-3.5-turbo-0613, text-danvinci-003, text-danvinci-002) and GPT-3 (davinci)

Source: generate_data/e2_generate_llm_results.py

#### aggregated_human_ratings.csv
An aggregated version of 'clean_human_ratings.csv', aggregated over every argument after filtering out unreliable MTurkers.

Source: generate_data/e2_preprocess_mturk_results.py

#### clean_human_ratings.csv
A cleaned version of 'raw_human_ratings.csv'.

Source: generate_data/e2_preprocess_mturk_results.py

#### raw_human_ratings.csv
Raw human ratings collected from our MTurk experiment.

Source: KR

#### unpaid_participants.csv
Contains MTurk UIDs for participants who were unpaid. Before analysis, we should exclude these participants. These participants have overlapping TIDs with other participants who were paid (also in this file).

Source: KR

# Pre experiment data files
Data used to run our MTurk/LLM experiments.

#### experiment_trials.csv
Trial (tid) seen by a participant (pid) in our MTurk experiment. This file was used to generate our MTurk experiment jsons. Does not include tutorial trials.

Source: Google Colab

#### control_trials.csv
Control trials that appears in experiment 2. There should be 4 trials for each single premise domain/conclusiontype split and 1 for each multi premise split. We only show participants 4 control trials; 3 single and 1 multi.

Source: handwritten

#### tutorial_trials.csv
Fruit argument tutorial trials used in MTurk experiment and 'T' prompt LLM experiments.

Source: handwritten

#### tutorial_trial_responses.csv
LLM responses to above fruit argument tutorial trials, saved so that we can use these as the prompt for 'T' prompt LLM experiments.

Source: generate_data/e2_generate_llm_results.py

### llm_prompts.csv
S3-C1-A1-Q1-O1-T style prompts for every argument used in the MTurk experiment, for every LLM under consideration.

Source: generate_data/e2_generate_llm_results.py
