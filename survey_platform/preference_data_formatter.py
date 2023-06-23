import pandas as pd
from tqdm.auto import tqdm

survey_results_pd = pd.read_csv("data/survey_results.csv")
survey_results_pd.head(10)

survey_results_formatted_df = survey_results_pd.copy()
survey_results_formatted_df["preferred"] = survey_results_formatted_df.apply(lambda row: row['option_a'] if row['decision'] == "Trajectory A" else row['option_b'], axis=1)
survey_results_formatted_df["not_preferred"] = survey_results_formatted_df.apply(lambda row: row['option_a'] if row['decision'] != "Trajectory A" else row['option_b'], axis=1)

prferred_df = pd.DataFrame() 
not_prferred_df = pd.DataFrame()
for idx, row in tqdm(survey_results_formatted_df.iterrows(), total=len(survey_results_formatted_df)):
    tmp_df = pd.read_csv(f"data/trajectories/{row['preferred']}")
    tmp_df["preference"] = 1
    prferred_df = prferred_df.append(tmp_df)
    tmp_df = pd.read_csv(f"data/trajectories/{row['not_preferred']}")
    tmp_df["preference"] = 0
    not_prferred_df = not_prferred_df.append(tmp_df)


preference_trajectory_df = pd.concat([prferred_df, not_prferred_df], axis=0)
preference_trajectory_df.to_csv("data/preference_trajectory.csv")
