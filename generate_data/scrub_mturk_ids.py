import pandas as pd
import os


def scrub_df(df: pd.DataFrame, id_map: dict) -> pd.DataFrame:
    df_scrubbed = df.copy(deep=True)
    df_scrubbed["uid"] = df_scrubbed["uid"].map(id_map)

    assert df_scrubbed.shape == df.shape
    assert len(set(df_scrubbed["uid"])) == len(set(df["uid"]))

    return df_scrubbed


if __name__ == "__main__":
    
    for experiment in ("experiment_1", "experiment_2"):

        df1 = pd.read_csv(f"../data/{experiment}/raw_human_ratings.csv")
        mturk_ids = df1["uid"].tolist()
        if os.path.exists(f"../data/{experiment}/clean_human_ratings.csv") and experiment == "experiment_1":
            df2 = pd.read_csv(f"../data/{experiment}/clean_human_ratings.csv")
            mturk_ids += df2["uid"].tolist()
        
        mturk_ids = list(set(mturk_ids))
        id_map = {mid: i for i, mid in enumerate(mturk_ids)}

        df1_scrubbed = scrub_df(df1, id_map)
        df1_scrubbed.to_csv(f"../data/{experiment}/raw_human_ratings.csv")

        if os.path.exists(f"../data/{experiment}/clean_human_ratings.csv") and experiment == "experiment_1":
            df2_scrubbed = scrub_df(df2, id_map)
            df2_scrubbed.to_csv(f"../data/{experiment}/clean_human_ratings.csv")