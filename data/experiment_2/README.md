#### aggregated_human_ratings.csv
An aggregated version of 'clean_human_ratings.csv', aggregated over every argument after filtering out unreliable participants.

#### clean_human_ratings.csv
A cleaned version of 'raw_human_ratings.csv'.

#### raw_human_ratings.csv
Raw human ratings collected from our MTurk experiment.

#### experiment_trials.csv
Trial (tid) seen by a participant (pid) in our MTurk experiment. This file was used to generate our MTurk experiment jsons. Does not include tutorial trials.

#### control_trials.csv
Control trials that appears in experiment 2. There should be 4 trials for each single premise domain/conclusiontype split and 1 for each multi premise split. We only show participants 4 control trials; 3 single and 1 multi.

#### unpaid_participants.csv
This file contains MTurk UIDs for participants who were unpaid. Before analysis, we should exclude these participants. These participants have overlapping TIDs with other participants who were paid (also in this file).
