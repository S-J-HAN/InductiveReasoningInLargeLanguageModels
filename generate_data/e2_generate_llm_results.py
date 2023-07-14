import llms
import prompts
import config
import helpers
import tqdm
import time
import os

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from typing import List, Dict, Tuple


def generate_tutorial_answers(
        prompt: prompts.Experiment2Prompt,
        llm_reasoners: List[llms.LLMReasoner]
    ) -> Dict[Tuple[str], str]:
    """Generate tutorial trial responses for a given list of LLMs"""

    FILE_PATH = f"{config.E2_DATA}/tutorial_trial_responses.csv"

    # Find LLMs that we don't already have responses for
    missing_reasoners = []
    if not os.path.exists(FILE_PATH):
        missing_reasoners = llm_reasoners
        existing_df = pd.DataFrame([], columns=["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner", "tutorial_prompt"])
    else:
        existing_df = pd.read_csv(FILE_PATH, index_col=0)
        missing_reasoners = [lr for lr in llm_reasoners if lr.name not in existing_df["llm_reasoner"].unique()]
    
    # Run response generation on LLMs that we don't already have response for
    tutorial_trial_df = pd.read_csv(f"{config.E2_DATA}/tutorial_trials.csv", index_col=0)
    tutorial_trial_df["premises"] = tutorial_trial_df["premises"].apply(eval)
    rows = []
    for csp, df in tqdm.tqdm(tutorial_trial_df.groupby(["conclusion_type", "is_single_premise"])):
        for llm_reasoner in missing_reasoners:
            print(llm_reasoner.name)

            conclusion_type, is_single_premise = csp
            df = df.sort_values(by="trial_num", ascending=True)

            is_completion_reasoner = "completion" in llm_reasoner.name

            if is_single_premise:
                num_facts = "only one supporting fact"
            else:
                num_facts = "two supporting facts"

            for parent_domain in list(config.PARENT_DOMAINS.values()):
                
                sp = prompt.SYSTEM.format(parent_domain)
                cp = prompt.CONTEXT.format(parent_domain,parent_domain)
                tp = prompt.TUTORIAL.format(num_facts)
                ap1 = prompt.ARGUMENT.format("\n".join([prompt.FACT.format(p) for p in df.iloc[0]["premises"]]), df.iloc[0]["conclusion"])
                qp = prompt.QUESTION
                op = prompt.OPTION
                if is_completion_reasoner:
                    prompt1 = sp + cp + " " + tp + "\n\n" + ap1 + "\n\n" + qp + " " + op
                else:
                    prompt1 = [
                        {"role": "system", "content": sp},
                        {"role": "user", "content": cp + " " + tp + "\n\n" + ap1 + "\n\n" + qp + " " + op}
                    ]

                llm_response1 = llm_reasoner.generate_response(prompt1)

                time.sleep(config.SLEEP_TIMES[llm_reasoner.vendor])

                ap2 = prompt.ARGUMENT.format("\n".join([prompt.FACT.format(p) for p in df.iloc[1]["premises"]]), df.iloc[1]["conclusion"])
                if is_completion_reasoner:
                    prompt2 = prompt1 + "\n" + llm_response1 + "\n\n" + ap2 + "\n\n" + qp + " " + op
                else:
                    prompt2 = prompt1 + [
                        {"role": "assistant", "content": llm_response1},
                        {"role": "user", "content": ap2 + "\n\n" + qp + " " + op}
                    ]

                llm_response2 = llm_reasoner.generate_response(prompt2)

                if is_completion_reasoner:
                    completed_tutorial_prompt = prompt2 + "\n" + llm_response2 + "\n\n" + prompt.SEGWAY + "\n\n"
                else:
                    completed_tutorial_prompt = prompt2 + [
                        {"role": "assistant", "content": llm_response2},
                        {"role": "user", "content": prompt.SEGWAY + "\n\n"}
                    ]

                rows.append([conclusion_type, is_single_premise, parent_domain, llm_reasoner.name, completed_tutorial_prompt])

                time.sleep(config.SLEEP_TIMES[llm_reasoner.vendor])
    
        tutorial_trial_response_df = pd.DataFrame(rows, columns=["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner", "tutorial_prompt"])
        output_df = pd.concat([existing_df, tutorial_trial_response_df], ignore_index=True)
        output_df = output_df.drop_duplicates(subset=["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner"]).reset_index(drop=True)
        output_df.to_csv(FILE_PATH)

    output_df["tutorial_prompt"] = output_df["tutorial_prompt"].apply(lambda x: eval(x) if x[0] == "[" else x)
    output_map = output_df.set_index(["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner"]).to_dict()["tutorial_prompt"]

    assert all(lr.name in set(output_df["llm_reasoner"].tolist()) for lr in llm_reasoners)
    assert output_df.shape[0] >= len(set(output_df["llm_reasoner"].tolist())) * 2 * 2 * 2
    assert output_df.shape[0] / len(set(output_df["llm_reasoner"].tolist())) == 2 * 2 * 2
    assert output_df.shape[0] == output_df.drop_duplicates(subset=["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner"]).shape[0]

    return output_map


def generate_prompt_df(
    tutorial_trial_response_map: Dict[Tuple[str], str],
    prompt: prompts.Experiment2Prompt,
    llm_reasoners: List[llms.LLMReasoner]
) -> pd.DataFrame:
    """Generate LLM prompts for every argument"""

    # Generate dataframe of prompts for each argument
    human_df = pd.read_csv(f"{config.E2_DATA}/aggregated_human_ratings.csv", index_col=0)
    human_df["premises"] = human_df["premises"].apply(lambda x: list(eval(x))).tolist()
    argument_df = human_df[["argument", "domain", "conclusion_type", "is_single_premise", "is_control", "premises", "conclusion"]]
    rows = []
    for _, row in argument_df.iterrows():
        for llm_reasoner in llm_reasoners:
            tp = tutorial_trial_response_map[(row["conclusion_type"], row["is_single_premise"], config.PARENT_DOMAINS[row["domain"]], llm_reasoner.name)]
            argument = prompts.Argument(row["premises"], row["conclusion"])
            llm_prompt = prompt.generate_prompt(argument, row["domain"], llm_reasoner.api_type == "completion", tp)
            rows.append((llm_reasoner.name, llm_reasoner.model) + tuple(row.values) + (llm_prompt,))
    prompt_df = pd.DataFrame(rows, columns=["llm_reasoner", "llm_model", "argument", "domain", "conclusion_type", "is_single_premise", "is_control", "premises", "conclusion", "prompt"])

    assert prompt_df.shape[0] == argument_df.shape[0] * len(llm_reasoners)
    assert set(prompt_df["argument"]) == set(argument_df["argument"])
    assert set(prompt_df["llm_reasoner"]) == set([lr.name for lr in llm_reasoners])
    assert set(prompt_df["llm_model"]) == set([lr.model for lr in llm_reasoners])

    prompt_df = prompt_df.sort_values(by=["llm_reasoner", "argument"]).reset_index(drop=True)
    prompt_df.to_csv(f"{config.E2_DATA}/llm_prompts.csv")

    return prompt_df


if __name__ == "__main__":

    prompt = prompts.Experiment2Prompt()

    llm_reasoners = [
        llms.OpenAICompletionReasoner("davinci"),
        llms.OpenAICompletionReasoner("text-davinci-001"),
        llms.OpenAICompletionReasoner("text-davinci-002"),
        llms.OpenAICompletionReasoner("text-davinci-003"),
        llms.OpenAIChatReasoner("gpt-3.5-turbo-0613"),
        llms.OpenAIChatReasoner("gpt-4-0314"),
    ]

    tutorial_trial_response_map = generate_tutorial_answers(prompt, llm_reasoners)
    prompt_df = generate_prompt_df(tutorial_trial_response_map, prompt, llm_reasoners)
    rating_df = helpers.generate_llm_ratings(prompt_df, llm_reasoners, f"{config.E2_DATA}/llm_ratings.csv", is_experiment_2=True)
        