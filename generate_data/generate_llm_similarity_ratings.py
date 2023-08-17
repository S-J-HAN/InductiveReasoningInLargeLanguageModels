from collections import defaultdict

import numpy as np
import pandas as pd

import tqdm
import openai
import os

import llms
import prompts
import helpers
import config

if __name__ == "__main__":

    domain_categories = helpers.load_map(f"{config.DEDEYNE_DATA}/domain_categories.json")

    # {"Mammals": [(Dogs, Cats), (Dogs, Horses), ...], "Birds": [...], ...}
    domain_category_pairs = defaultdict(list)
    for domain in domain_categories:
        categories = domain_categories[domain]
        for category in categories:
            other_categories = [c for c in categories if c != category]
            for other_category in other_categories:
                domain_category_pairs[domain].append((category, other_category))

    prompt = prompts.SimilarityPrompt()

    llm_reasoners = [
        llms.OpenAIChatReasoner("gpt-3.5-turbo-0613"),
        llms.OpenAIChatReasoner("gpt-4-0314"),
        llms.OpenAICompletionReasoner("text-davinci-003"),
    ]

    # Generate prompts
    rows = []
    for domain, category_pairs in domain_category_pairs.items():
        for c1, c2 in category_pairs:
            for llm_reasoner in llm_reasoners:
                p = prompt.generate_prompt(domain, c1, c2, llm_reasoner.api_type == "completion")
                rows.append((llm_reasoner.name, llm_reasoner.model, domain, c1, c2, p))
    prompt_df = pd.DataFrame(rows, columns=["llm_reasoner", "llm_model", "domain", "category1", "category2", "prompt"])
    prompt_df = prompt_df.sort_values(by="llm_model").reset_index(drop=True)
    prompt_df["argument"] = prompt_df.index.tolist()
    prompt_df.to_csv(f"{config.SIMILARITY_DATA}/llm_prompts.csv")

    # Generate completions
    ratings_df = helpers.generate_llm_ratings(prompt_df, llm_reasoners, "", True)

    # Save ratings into json files
    for llm_reasoner in llm_reasoners:
        model_df = ratings_df[ratings_df["llm_model"] == llm_reasoner.model]

        assert all(set(model_df[c]) == set(ratings_df[c]) for c in ["category1", "category2", "domain"])

        model_ratings = model_df.set_index(["domain", "category1", "category2"]).to_dict()["llm_rating"]
        model_similarity_map = {c: {} for c in model_df["category1"].unique()}
        for domain in model_df["domain"].unique():
            categories = domain_categories[domain]
            for category in categories:
                other_categories = [c for c in categories if c != category]
                for other_category in other_categories:
                    k1, k2 = (domain, category, other_category), (domain, other_category, category)
                    model_similarity_map[category][other_category] = np.mean([model_ratings[k1], model_ratings[k2]])
        
        helpers.save_map(model_similarity_map, f"{config.SIMILARITY_DATA}/{llm_reasoner.model}_similarity_map.json")

    
    # Generate embeddings based similarity ratings
    embeddings = {}
    openai.api_key = os.environ["OPENAI"]
    for domain, dc in tqdm.tqdm(domain_categories.items()):
        r = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=dc
        )
        for i, c in enumerate(dc):
            embeddings[c] = r["data"][i]["embedding"]
    helpers.save_map(embeddings, f"{config.SIMILARITY_DATA}/openai_category_embeddings.json")

    cosine_similarities = {}
    for key1, vec1 in tqdm.tqdm(embeddings.items()):
        cosine_similarities[key1] = {}
        for key2, vec2 in embeddings.items():
            if key1 != key2:
                cosine_similarities[key1][key2] = helpers.calculate_cosine_similarity(vec1, vec2)
    helpers.save_map(cosine_similarities, f"{config.SIMILARITY_DATA}/openai_embedding_similarity_map.json")
        