import llms
import prompts
import config
import tqdm
import time
import os

import pandas as pd

from typing import List


def generate_tutorial_answers(llm_reasoners: List[llms.LLMReasoner]) -> pd.DataFrame:
    """Generate tutorial trial responses for a given list of LLMs"""

    FILE_PATH = f"{config.E2_DATA}/tutorial_trial_responses.csv"

    prompt = prompts.Experiment2Prompt()

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
                ip = prompt.CONTEXT.format(parent_domain,parent_domain,num_facts)
                ap1 = prompt.ARGUMENT.format("\n".join([prompt.FACT.format(p) for p in df.iloc[0]["premises"]]), df.iloc[0]["conclusion"])
                q = prompt.QUESTION
                o = prompt.OPTION
                if is_completion_reasoner:
                    prompt1 = sp + ip + ap1 + q + o
                else:
                    prompt1 = [
                        {"role": "system", "content": sp},
                        {"role": "user", "content": ip + ap1 + q + o}
                    ]

                llm_response1 = llm_reasoner.generate_response(prompt1)

                time.sleep(config.SLEEP_TIMES[llm_reasoner.vendor])

                ap2 = prompt.ARGUMENT.format("\n".join([prompt.FACT.format(p) for p in df.iloc[1]["premises"]]), df.iloc[1]["conclusion"])
                if is_completion_reasoner:
                    prompt2 = prompt1 + "\n" + llm_response1 + "\n\n" + ap2 + q + o
                else:
                    prompt2 = prompt1 + [
                        {"role": "assistant", "content": llm_response1},
                        {"role": "user", "content": ap2 + q + o}
                    ]

                llm_response2 = llm_reasoner.generate_response(prompt2)

                if is_completion_reasoner:
                    completed_tutorial_prompt = prompt2 + "\n" + llm_response2 + "\n\n" + prompt.SEGWAY
                else:
                    completed_tutorial_prompt = prompt2 + [
                        {"role": "assistant", "content": llm_response2},
                        {"role": "user", "content": prompt.SEGWAY}
                    ]

                rows.append([conclusion_type, is_single_premise, parent_domain, llm_reasoner.name, completed_tutorial_prompt])

                time.sleep(config.SLEEP_TIMES[llm_reasoner.vendor])
    
        tutorial_trial_response_df = pd.DataFrame(rows, columns=["conclusion_type", "is_single_premise", "parent_domain", "llm_reasoner", "tutorial_prompt"])
        output_df = pd.concat([existing_df, tutorial_trial_response_df], ignore_index=True)

        output_df.to_csv(FILE_PATH)

    return output_df
    


if __name__ == "__main__":

    llm_reasoners = [
        llms.CohereCompletionReasoner("command"),
        llms.OpenAIChatReasoner("gpt-3.5-turbo"),
        llms.OpenAIChatReasoner("gpt-3.5-turbo-0613"),
        llms.OpenAIChatReasoner("gpt-4"),
        llms.OpenAIChatReasoner("gpt-4-0314"),
        llms.OpenAICompletionReasoner("text-davinci-003")
    ]

    tutorial_trial_response_df = generate_tutorial_answers(llm_reasoners)

    assert all(lr.name in set(tutorial_trial_response_df["llm_reasoner"].tolist()) for lr in llm_reasoners)
    assert tutorial_trial_response_df.shape[0] == len(set(tutorial_trial_response_df["llm_reasoner"].tolist())) * 2 * 2 * 2
