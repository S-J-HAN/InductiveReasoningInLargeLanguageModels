from typing import List, Dict

import numpy as np
import pandas as pd

import helpers
import config


def get_similarity(
    a: str, 
    b: str, 
    sim_map: Dict
) -> float:
    """Return similarity between two categories, a and b"""
    if a in sim_map and b in sim_map[a]:
        return sim_map[a][b]
    elif b in sim_map and a in sim_map[b]:
        return sim_map[b][a]
    else:
        return 0.0
  

def scm(
    premises: List[str], 
    c: str, 
    all_c: List[str], 
    sim_map: Dict[str, Dict[str,float]], 
    specific: bool = True, 
    alpha: float = 0.5
) -> float:
    """Calculate SCM score of a given argument"""

    if not specific:
        conclusion_categories_1 = all_c
    else:
        conclusion_categories_1 = [c]
    a = np.mean([
            np.max([
                get_similarity(p, c_cat, sim_map)
                for p in premises if p
            ])
            for c_cat in conclusion_categories_1
    ])
        
    # calculate b
    conclusion_categories_2 = all_c
    b = np.mean([
            np.max([
                get_similarity(p, c_cat, sim_map)
                for p in premises if p
            ])
            for c_cat in conclusion_categories_2
    ]) 
    
    return alpha*a + (1-alpha)*b


if __name__ == "__main__":

    domain_categories = helpers.load_map(f"{config.DEDEYNE_DATA}/domain_categories.json")
    similarity_maps = {
        r: helpers.load_map(f"{config.SIMILARITY_DATA}/{r}_similarity_map.json") 
        for r in ["gpt3", "gpt4", "human"]
    }

    # Flatten existing human similarity map, which is split by domains
    human_similarity_map = {}
    for k,v in similarity_maps["human"].items():
        for k1,v1 in v.items():
            human_similarity_map[k1] = v1
    similarity_maps["human"] = human_similarity_map

    llm_rating_df = pd.read_csv(f"{config.E2_DATA}/llm_ratings.csv", index_col=0)
    llm_rating_df["premises"] = llm_rating_df["premises"].apply(eval)

    rows = []
    for argument, argument_df in llm_rating_df.groupby("argument"):
        
        r = argument_df.iloc[0]
        domain, conclusion_type, premises, conclusion, is_single_premise = r["domain"], r["conclusion_type"], r["premises"], r["conclusion"], r["is_single_premise"]
        
        gpt3_rating = argument_df[argument_df["llm_model"] == "text-davinci-003"]["llm_rating"].iloc[0]
        gpt35_rating = argument_df[argument_df["llm_model"] == "gpt-3.5-turbo-0613"]["llm_rating"].iloc[0]
        gpt4_rating = argument_df[argument_df["llm_model"] == "gpt-4-0314"]["llm_rating"].iloc[0]
        
        rows.append((argument, domain, conclusion_type, is_single_premise, premises, conclusion, gpt3_rating, gpt35_rating, gpt4_rating))

    model_df = pd.DataFrame(rows, columns=["argument", "domain", "conclusion_type", "is_single_premise", "premises", "conclusion", "gpt3_rating", "gpt3.5_rating", "gpt4_rating"])

    # MaxSim
    for agent, agent_similarity_map in similarity_maps.items():
        model_df[f"{agent}_maxsim"] = [
            np.max([
                agent_similarity_map[p][row["conclusion"]]
                for p in row["premises"]
            ])
            if row["conclusion_type"] == "Specific"
            else None
            for _, row in model_df.iterrows()
        ]

    # MeanSim
    for agent, agent_similarity_map in similarity_maps.items():
        model_df[f"{agent}_meansim"] = [
            np.mean([
                agent_similarity_map[p][row["conclusion"]]
                for p in row["premises"]
            ])
            if row["conclusion_type"] == "Specific"
            else None
            for _, row in model_df.iterrows()
        ]

    # SCM
    ALPHA = 0.5
    scm_score_maps = {agent: {} for agent in similarity_maps}
    for cdsp, tdf in model_df.groupby(["conclusion_type", "domain", "is_single_premise"]):
        conclusion_type, domain, is_single_premise = cdsp
        for agent, agent_similarity_map in similarity_maps.items():

            scm_scores = []
            for _, row in tdf.iterrows():
                try:
                    scm_score = scm(
                        row["premises"], 
                        row["conclusion"], 
                        domain_categories[domain], 
                        agent_similarity_map, 
                        row["conclusion_type"] == "Specific",
                        alpha=ALPHA
                    )
                    scm_scores.append(scm_score)
                except:
                    scm_scores.append(None)

            for i, row in tdf.reset_index(drop=True).iterrows():
                scm_score_maps[agent][row["argument"]] = scm_scores[i]

    for agent, scm_score_map in scm_score_maps.items():
        model_df[f"{agent}_scm"] = [scm_score_map[row["argument"]] for _, row in model_df.iterrows()]

    assert len(model_df) == len(llm_rating_df) / len(set(llm_rating_df["llm_model"]))
    assert set(model_df["argument"]) == set(llm_rating_df["argument"])
    assert len(model_df[model_df["conclusion_type"] == "Specific"]) == len(model_df[model_df["conclusion_type"] == "Specific"].dropna(subset=["gpt3_maxsim", "gpt3_meansim", "gpt4_maxsim", "gpt4_meansim", "human_maxsim", "human_meansim"]))
    assert len(model_df) == len(model_df.dropna(subset=["gpt3_scm", "gpt4_scm", "human_scm"]))

    model_df.to_csv(f"{config.E2_DATA}/model_ratings.csv")