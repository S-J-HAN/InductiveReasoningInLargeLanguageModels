import warnings
warnings.filterwarnings("ignore")

import llms
import prompts
import config
import helpers

import pandas as pd

from typing import List


def generate_prompt_df(
    prompt: prompts.Experiment1Prompt,
    llm_reasoners: List[llms.LLMReasoner]
) -> pd.DataFrame:
    """Generate LLM prompts for every argument"""

    # Generate dataframe of prompts for each argument
    human_df = pd.read_csv(f"{config.E1_DATA}/aggregated_human_ratings.csv", index_col=0)
    human_df["stronger_arg_premises"] = human_df["stronger_arg_premises"].apply(lambda x: list(eval(x))).tolist()
    human_df["weaker_arg_premises"] = human_df["weaker_arg_premises"].apply(lambda x: list(eval(x))).tolist()
    argument_df = human_df[["argpair", "phenomenon", "domain", "stronger_arg_premises", "stronger_arg_conclusion", "weaker_arg_premises", "weaker_arg_conclusion", "is_control", "is_osherson", "is_weaker_arg_shown_first"]]
    rows = []
    for _, row in argument_df.iterrows():
        for llm_reasoner in llm_reasoners:
            stronger_argument = prompts.Argument(row["stronger_arg_premises"], row["stronger_arg_conclusion"])
            weaker_argument = prompts.Argument(row["weaker_arg_premises"], row["weaker_arg_conclusion"])
            if row["show_weaker_arg_first"]:
                argument_a, argument_b = weaker_argument, stronger_argument
            else:
                argument_a, argument_b = stronger_argument, weaker_argument
            llm_prompt = prompt.generate_prompt(argument_a, argument_b, row["domain"], llm_reasoner.api_type == "completion")
            rows.append((llm_reasoner.name, llm_reasoner.model) + tuple(row.values) + (llm_prompt,))
    prompt_df = pd.DataFrame(rows, columns=["llm_reasoner", "llm_model", "argpair", "phenomenon", "domain", "stronger_arg_premises", "stronger_arg_conclusion", "weaker_arg_premises", "weaker_arg_conclusion", "is_control", "is_osherson", "weaker_arg_shown_first", "prompt"])

    assert prompt_df.shape[0] == argument_df.shape[0] * len(llm_reasoners)
    assert set(prompt_df["argpair"]) == set(argument_df["argpair"])
    assert set(prompt_df["llm_reasoner"]) == set([lr.name for lr in llm_reasoners])
    assert set(prompt_df["llm_model"]) == set([lr.model for lr in llm_reasoners])

    prompt_df = prompt_df.sort_values(by=["llm_reasoner", "argpair"], ascending=False).reset_index(drop=True)
    prompt_df.to_csv(f"{config.E1_DATA}/llm_prompts.csv")

    return prompt_df


if __name__ == "__main__":

    prompt = prompts.Experiment1Prompt()

    llm_reasoners = [
        llms.OpenAICompletionReasoner("davinci"),
        llms.OpenAICompletionReasoner("text-davinci-001"),
        llms.OpenAICompletionReasoner("text-davinci-002"),
        llms.OpenAICompletionReasoner("text-davinci-003"),
        llms.OpenAIChatReasoner("gpt-3.5-turbo-0613"),
        llms.OpenAIChatReasoner("gpt-4-0314"),
    ]

    prompt_df = generate_prompt_df(prompt, llm_reasoners)
    rating_df = helpers.generate_llm_ratings(prompt_df, llm_reasoners, f"{config.E1_DATA}/llm_ratings.csv", is_experiment_2=False)

    # Set rating column so that lower number always means the stronger SCM argument was rated as stronger
    rating_df["llm_rating"] = [5 - row["llm_rating"] if row["llm_rating"] and row["is_weaker_arg_shown_first"] else row["llm_rating"] for _, row in rating_df.iterrows()]
    rating_df.to_csv(f"{config.E1_DATA}/llm_ratings.csv")
