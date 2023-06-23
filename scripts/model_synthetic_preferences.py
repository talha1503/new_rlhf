import os
import ast
import numpy as np
import pandas as pd
import random
import pickle
from tqdm.auto import tqdm
import click


# Paths
## Input
# trajectories_folder = "./data/trajectories/"
# preference_model_path = "./models/logreg_fair_model.sav"
# preference_model_path = "./models/logreg_greedy_model.sav"

## Output
# preferences_output_path = "./data/fair_syntehtic_preferences.csv"
# preferences_output_path = "./data/greedy_syntehtic_preferences.csv"

    
def expand_credit_score_distribution_column(df, distribution_col):
    return pd.DataFrame(df[distribution_col].to_list(), columns=[f"{distribution_col}_{i}" for i in range(7)])
    
def format_input_data(input_df, meta_cols):
    df = input_df.copy()
    df = df.reset_index(drop=True)
    df["credit_score_distribution-group_1"] = df["credit_score_distribution-group_1"].apply(lambda x: ast.literal_eval(x))
    df["credit_score_distribution-group_2"] = df["credit_score_distribution-group_2"].apply(lambda x: ast.literal_eval(x))

    credit_score_distribution_1 = expand_credit_score_distribution_column(df, "credit_score_distribution-group_1")
    credit_score_distribution_2 = expand_credit_score_distribution_column(df, "credit_score_distribution-group_2")
    df = pd.concat([df, credit_score_distribution_1, credit_score_distribution_2], axis=1)
    
    drop_meta_cols = [c for c in meta_cols if c in df.columns]
    df = df.drop(drop_meta_cols+["credit_score_distribution-group_1", "credit_score_distribution-group_2"], axis=1)
    return df
    
def generate_synthetic_selections(preference_model, meta_cols, trajectories_folder, n_decisions=100):
    """ 
    Args:
        preference_model: A model which would output a preference given the input features specified.
        trajectories_folder: Path to the folder with trajectories for sampling.
        n_decisions: The number of synthetic decisions required.
        
    Return:
        pd.DataFrame: A dataframe with the different simulated options and synthetic preferences.
    """
    trajectories = [i for i in os.listdir(trajectories_folder) if '.csv' in i]
    columns_format = ["acceptance_rate-group_1","acceptance_rate-group_2",
                      "default_rate-group_1","default_rate-group_2",
                      "average_credit_score-group_1","average_credit_score-group_2",
                      "credit_score_distribution-group_1","credit_score_distribution-group_2",
                      "applicant_group_membership",
                      "applicant_credit_group",
                      "agent_action"
                     ]

    synthetic_preference_df = pd.DataFrame()
    for _ in tqdm(range(n_decisions)):
        # Randomly select two trajectories for making preference
        trajectory_a, trajectory_b = random.sample(trajectories, 2)
        trajectory_a_df = pd.read_csv(f"{trajectories_folder}/{trajectory_a}", index_col=[0])
        trajectory_b_df = pd.read_csv(f"{trajectories_folder}/{trajectory_b}", index_col=[0])
        
        # +1 because the group membership is [0, 1] but the feature names consist of group_1 and group_2.
        trajectory_a_df['applicant_group_membership'] = trajectory_a_df['applicant_group_membership'] + 1
        trajectory_b_df['applicant_group_membership'] = trajectory_b_df['applicant_group_membership'] + 1

        # Format trajectories and sample options for decisioning
        trajectory_a_df_filtered = trajectory_a_df[columns_format]
        trajectory_b_df_filtered = trajectory_b_df[columns_format]

        # Sample 1 time-step for each segment
        option_a = trajectory_a_df_filtered.sample(1).reset_index(drop=True)
        option_b = trajectory_b_df_filtered.sample(1).reset_index(drop=True)
        
        feature_a = option_a.iloc[:, :-1]
        feature_b = option_b.iloc[:, :-1]
        action_a = option_a.iloc[:, -1].values
        action_b = option_b.iloc[:, -1].values
        
        # Format the input for the preference model
        a_X = format_input_data(feature_a, meta_cols)
        b_X = format_input_data(feature_b, meta_cols)
        # Make synthetic decision        
        pref_a = preference_model.predict(a_X)
        pref_b = preference_model.predict(b_X)
        
        # Preference distribution for A and B, with:
        # [1,0] -> Prefer A over B; [0,1] -> Prefer B over A
        # [1,1] -> Equally good; [0,0] -> Incomparable
        preference_distribution = [0, 0]
        if pref_a == action_a:
            preference_distribution[0] = 1
        if pref_b == action_b:
            preference_distribution[1] = 1
            
        option_a.columns = [f"option_a-{i}" for i in option_a.columns]
        option_b.columns = [f"option_b-{i}" for i in option_b.columns]
        tmp_df = pd.concat([option_a, option_b], axis=1)
        tmp_df['preference'] = str(preference_distribution)
        synthetic_preference_df = synthetic_preference_df.append(tmp_df)

    synthetic_preference_df = synthetic_preference_df.reset_index(drop=True)
    return synthetic_preference_df
    
    
@click.command()
@click.options("--preference_model_path", default="./models/logreg_fair_model.sav")
@click.options("--trajectories_folder", default="./data/trajectories/")
@click.options("--preferences_output_path", default="./data/fair_syntehtic_preferences.csv")
@click.options("--n_decisions", default=500)
def generate_synthetic_preferences(preference_model_path, 
                                   trajectories_folder,
                                   preferences_output_path,
                                   n_decisions
                                  ): 

    meta_cols = ["Timestep", "sample_id"]

    # Load model
    preference_model = pickle.load(open(preference_model_path, 'rb'))

    # Generate synthetic decisions
    synthetic_preference_df = generate_synthetic_selections(preference_model, meta_cols, trajectories_folder, n_decisions=n_decisions)
    # Export results
    synthetic_preference_df.to_csv(preferences_output_path, index=False)

    
if __name__ == "__main__":
    generate_synthetic_preferences()
    