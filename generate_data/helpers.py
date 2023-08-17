from typing import Optional, List
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

import json
import multiprocessing
import tqdm

import llms


def save_map(m: object, filename: str) -> None:
  with open(filename, 'w') as f:
    json.dump(m, f)


def load_map(filename: str) -> None:
  with open(filename, 'r') as f:
    return json.load(f)
  

def get_rating(prompt: str, is_experiment_2: bool, llm_reasoner: llms.LLMReasoner) -> Optional[llms.LLMRating]:
  """Helper function for multiprocessing, used below"""
  return llm_reasoner.generate_rating(prompt, is_experiment_2)


def generate_llm_ratings(
    prompt_df: pd.DataFrame,
    llm_reasoners: List[llms.LLMReasoner],
    output_path: str,
    is_experiment_2: bool,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Generates LLM ratings for a given prompt dataframe"""

    rating_df = prompt_df.copy(deep=True).sort_values(by=["llm_reasoner", "argpair" if not is_experiment_2 else "argument"])
    llm_reasoner_map = {llm_reasoner.name: llm_reasoner for llm_reasoner in llm_reasoners}
    
    raw_completions, ratings = [], []
    for b in tqdm.tqdm(range(0,len(rating_df),batch_size)):
        batch_df = rating_df.iloc[b:b+batch_size]
        inputs = [(row["prompt"], is_experiment_2, llm_reasoner_map[row["llm_reasoner"]]) for _, row in batch_df.iterrows()]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            outputs = pool.starmap(get_rating, inputs)
            raw_completions += [x.raw_completion if x else None for x in outputs]
            ratings += [x.parsed_rating if x else None for x in outputs]
        rating_df["llm_raw_completion"] = raw_completions + [None]*(len(rating_df)-len(raw_completions))
        rating_df["llm_rating"] = ratings + [None]*(len(rating_df)-len(ratings))
        rating_df = rating_df.sort_values(by=["llm_reasoner", "argpair" if not is_experiment_2 else "argument"]).reset_index(drop=True)

        if output_path:
          rating_df.to_csv(output_path)

    assert len(prompt_df) == len(rating_df)
    
    return rating_df


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]