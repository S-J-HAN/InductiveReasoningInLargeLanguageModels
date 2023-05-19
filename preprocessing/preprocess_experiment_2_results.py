import pandas as pd

import config
import tqdm


def determine_if_row_is_control(row: pd.Series, control_df: pd.DataFrame) -> bool:
    """Given a row in the MTurk results dataframe, determine if this is a control trial"""

    if "All" in row["premises"][0]:
        return True

    control_rows = control_df[(control_df["conclusion_type"] == row["conclusion_type"]) & (control_df["domain"] == row["domain"]) & (control_df["is_single_premise"] == row["is_single_premise"])]
    
    for _, control_row in control_rows.iterrows():
        control_premise = tuple(control_row["premises"])
        control_conclusion = control_row["conclusion"]
        if tuple(control_premise) == row["premises"] and control_conclusion == row["conclusion"]:
            return True

    return False


def identify_participants_to_cut(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify participants that we should exclude from our analysis.
    
    'Light cut': participants who got more than one control trial incorrect.
    'Medium cut': Same as above, or those who gave at least 12 trials the exact same rating.
    'Hard cut': Same as above, or those who gave at least 8 trials the exact same rating.
    """

    rows = []
    for pid, pid_df in df.groupby("pid"):

        num_incorrect_controls = pid_df[(pid_df["is_control"]) & (pid_df["rating"] < 50)].shape[0]
        ratings_range = pid_df["rating"].max() - pid_df["rating"].min()
        ratings_mode = pid_df["rating"].value_counts().max()
        ratings_std = pid_df["rating"].std()

        rows.append((pid, num_incorrect_controls, ratings_range, ratings_mode, ratings_std))

    pid_cut_df = pd.DataFrame(rows, columns=["pid", "num_incorrect_controls", "ratings_range", "ratings_mode", "ratings_std"])
    pid_cut_df["light_cut"] = [row["num_incorrect_controls"] > 1 for _, row in pid_cut_df.iterrows()]
    pid_cut_df["medium_cut"] = [row["ratings_mode"] >= 12 or row["num_incorrect_controls"] > 1 for _, row in pid_cut_df.iterrows()]
    pid_cut_df["hard_cut"] = [row["ratings_mode"] >= 8 or row["num_incorrect_controls"] > 1 for _, row in pid_cut_df.iterrows()]

    output_df = df.merge(pid_cut_df[["pid", "light_cut", "medium_cut", "hard_cut"]], on="pid")
    assert output_df.shape[0] == df.shape[0]

    for cut_type in ("light_cut", "medium_cut", "hard_cut"):
        print(f"Number of {cut_type} participants: {pid_cut_df[pid_cut_df[cut_type]].shape[0]}/{pid_cut_df.shape[0]}")
    print()

    return output_df


def preprocess_experiment_2_results() -> None:
    """
    Preprocesses and saves human ratings dataframes for experiment 2
    """

    # Load original experiment split labels
    experiment_df = pd.read_csv(f"{config.DATA}/experiment_trials.csv", index_col=0)
    experiment_df["pid"] = experiment_df["pid"].apply(str)
    experiment_df["tid"] = experiment_df["tid"].apply(str)
    experiment_df["premises"] = experiment_df["premises"].apply(lambda x: tuple(eval(x)))
    experiment_df = experiment_df[experiment_df["is_osherson"] == 0].reset_index(drop=True)

    # Load human ratings
    human_df = pd.read_csv(f"{config.DATA}/raw_human_ratings.csv")
    human_df["pid"] = human_df["tid"].apply(lambda x: x.split("participant")[-1])
    human_df["tid"] = human_df["trialId"].apply(lambda x: x.replace("tc", ""))
    human_df["premises"] = human_df["premises0"].apply(lambda x: tuple(x.split(":")))
    human_df["conclusion"] = human_df["conclusion0"]
    human_df["conclusion_type"] = human_df["conclusionType"].apply(lambda x: x.capitalize())

    # Drop participants who were not paid
    unpaid_participants_df = pd.read_csv(f"{config.DATA}/unpaid_participants.csv", index_col=0)
    drop_uids = unpaid_participants_df[~unpaid_participants_df["paid"]]["uid"].tolist()
    human_df = human_df[~human_df["uid"].isin(drop_uids)].reset_index(drop=True)

    # Join human_df and experiment_df, keeping experiment_df's split labels
    df = human_df.merge(experiment_df, on=["pid", "tid", "conclusion_type", "premises", "conclusion"])
    df["is_single_premise"] = df["premises"].apply(lambda x: len(x) == 1)
    df = df[["pid", "tid", "domain", "conclusion_type", "is_single_premise", "premises", "conclusion", "rating"]].sort_values(by=["pid", "tid"]).reset_index(drop=True)

    # Do some checks
    print(f"Merged df nrows: {df.shape[0]}, human_df nrows: {human_df.shape[0]}, experiment_df nrows: {experiment_df.shape[0]}")
    print(f"Total number of PIDs: {len(human_df['pid'].unique())}")
    missing_pids = set(experiment_df["pid"]).difference(set(human_df["pid"]))
    print(f"Missing PIDs that are in experiment_df but not human_df: {missing_pids}")
    assert df.shape[0] == human_df.shape[0] == experiment_df[~experiment_df["pid"].isin(missing_pids)].shape[0]
    assert len(experiment_df["pid"].unique()) == len(df["pid"].unique()) + len(missing_pids)
    assert all(gdf.shape[0] == 38 for _,gdf in df.groupby("pid"))
    print()

    # Label control trials
    control_df = pd.read_csv(f"{config.DATA}/control_trials.csv", index_col=0)
    control_df["premises"] = control_df["premises"].apply(lambda x: tuple(eval(x)))
    print("Labelling trials as control or not control...")
    df["is_control"] = [determine_if_row_is_control(row, control_df) for _, row in tqdm.tqdm(df.iterrows())]
    assert all(p == 4 for p in df[df["is_control"]]["pid"].value_counts().tolist())
    assert df[df["is_control"]]["pid"].value_counts().shape[0] == len(df["pid"].unique())
    print()

    df = identify_participants_to_cut(df)
    df["argument"] = [(row["premises"], row["conclusion"]) for _, row in df.iterrows()]
    df = df[["pid", "tid",  "argument", "domain", "conclusion_type", "is_single_premise", "is_control", "premises", "conclusion", "rating", "light_cut", "medium_cut", "hard_cut"]]
    df.to_csv(f"{config.DATA}/clean_human_ratings.csv")

    # Exclude participants at the light cut level
    df = df[~df["light_cut"]]
    print(f"Number of participants left after light cut: {len(df['pid'].unique())}")

    # Get argument rankings at a per participant, premise number level
    df["ratings_rank"] = df.groupby(["pid", "is_single_premise"])['rating'].rank(pct="True", ascending=True)

    # Aggregate ratings and rankings across participants
    rows = []
    argument_labels = ["argument", "domain", "conclusion_type", "is_single_premise", "is_control", "premises", "conclusion"]
    for al, arg_df in df.groupby(argument_labels):

        avg_rank = arg_df["ratings_rank"].mean()
        avg_rating = arg_df["rating"].mean()

        num_ratings = arg_df.shape[0]

        rows.append(al + (avg_rating, avg_rank, num_ratings))
        
    aggregated_df = pd.DataFrame(rows, columns=argument_labels + ["average_rating", "average_ranking", "num_ratings"])
    aggregated_df = aggregated_df[~aggregated_df["is_control"]].reset_index(drop=True)

    # Check that number of arguments is correct and all are unique
    for split, split_df in aggregated_df.groupby(["domain", "conclusion_type", "is_single_premise"]):
        _, conclusion_type, is_single_premise = split
        if conclusion_type == "General":
            assert split_df.shape[0] == 24 if is_single_premise else split_df.shape[0] == 100
        else:
            assert split_df.shape[0] > 100 if is_single_premise else split_df.shape[0] == 100
        assert len(set(split_df["argument"].tolist())) == split_df.shape[0]
        
    # Print statistics on number of arguments per split
    print("Number of ratings per argument per split")
    for split, split_df in aggregated_df.groupby(["domain", "conclusion_type", "is_single_premise"]):
        r = split_df["num_ratings"]
        minr, maxr, meanr, sdr = r.min(), r.max(), r.mean(), r.std()
        print(f"    {split}: min - {minr}, max - {maxr}, mean - {meanr:.2f}, sd - {sdr:.2f}")

    aggregated_df.to_csv(f"{config.DATA}/aggregated_human_ratings.csv")


if __name__ == "__main__":
    preprocess_experiment_2_results()
