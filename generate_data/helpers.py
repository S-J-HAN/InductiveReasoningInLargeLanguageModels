from typing import Tuple, Any, Optional, List
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

import json
import multiprocessing
import tqdm

import llms
import helpers


def save_map(m: object, filename: str) -> None:
  with open(filename, 'w') as f:
    json.dump(m, f)

def load_map(filename: str) -> None:
  with open(filename, 'r') as f:
    return json.load(f)
  

def get_rating(prompt: str, llm_reasoner: llms.LLMReasoner) -> Tuple[Any, Optional[float]]:
  """Helper function for multiprocessing, used below"""
  return llm_reasoner.generate_rating(prompt)


def generate_llm_ratings(
    prompt_df: pd.DataFrame,
    llm_reasoners: List[llms.LLMReasoner],
    output_path: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Generates LLM ratings for a given prompt dataframe"""

    rating_df = prompt_df.copy(deep=True).sort_values(by="llm_reasoner")
    llm_reasoner_map = {llm_reasoner.name: llm_reasoner for llm_reasoner in llm_reasoners}
    
    raw_completions, ratings = [], []
    for b in tqdm.tqdm(range(0,len(rating_df),batch_size)):
        batch_df = rating_df.iloc[b:b+batch_size]
        inputs = [(row["prompt"], llm_reasoner_map[row["llm_reasoner"]]) for _, row in batch_df.iterrows()]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            outputs = pool.starmap(helpers.get_rating, inputs)
            raw_completions += [x[0] for x in outputs]
            ratings += [x[1] for x in outputs]
        print(len([x[0] for x in outputs if x[0]]))
        rating_df["llm_raw_completion"] = raw_completions + [None]*(len(rating_df)-len(raw_completions))
        rating_df["llm_rating"] = ratings + [None]*(len(rating_df)-len(ratings))
        rating_df.to_csv(output_path)

    assert len(prompt_df) == len(rating_df)
    
    return rating_df


def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]