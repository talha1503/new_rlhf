import click
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import torch
from source.evaluate_models import evaluate_model
from source.losses import preference_loss_function
from source.mlp import MLP
from source.training import train_reward_model


state_action_features = ['default_rate-group_1', 'default_rate-group_2', 
                         'acceptance_rate-group_1', 'acceptance_rate-group_2', 
                         'average_credit_score-group_2', 'average_credit_score-group_1',
                         'applicant_credit_group', 'applicant_group_membership', 'agent_action']


def n_sample_trajectory_df(input_df, n_sample_points = 20):
    trajectory_df = input_df.copy()
    if len(trajectory_df)//2 > n_sample_points:
        sample_step_size = len(trajectory_df) // n_sample_points
        trajectory_df = trajectory_df.reset_index(drop=True)
        sample_index = [i for i in range(len(trajectory_df)) if i%sample_step_size == 0]
        trajectory_df_sampled = trajectory_df[trajectory_df.index.isin(sample_index)]
        return trajectory_df_sampled
    else:
        return trajectory_df


@click.command()
@click.option('--preference_data_path', default="./data/synthetic_decisions.csv")
@click.option('--model_eval_path', default="./data/reward_model_eval.json")
@click.option('--num_epochs', default=20)
@click.option('--reward_model_name', default="reward_model")
@click.option('--save_dir', default="./models/")
@click.option('--training_curve_path', default="./data/reward_model_loss.png")
@click.option('--n_sample_points', default=50)
def train(preference_data_path,
                       model_eval_path,
                       num_epochs,
                       reward_model_name,
                       save_dir,
                       training_curve_path,
                       n_sample_points
                      ):

    target = "target"
    
    # Data formatting
    preferences_df = pd.read_csv(preference_data_path)
    
    # Extracting the trajectory data given the filenames from compared history
    preferences_df_formatted = pd.DataFrame()
    modelling_df = preferences_df.copy()
    for idx, row in tqdm(modelling_df.iterrows(), total=len(modelling_df)):

        option_a_file = row['Trajectory_A']
        option_b_file = row['Trajectory_B']

        tmp_df_a = pd.read_csv(f"./data/trajectories/{option_a_file}", index_col=[0])
        tmp_df_b = pd.read_csv(f"./data/trajectories/{option_b_file}", index_col=[0])

        tmp_df_a_sampled = n_sample_trajectory_df(tmp_df_a, n_sample_points=n_sample_points)
        tmp_df_b_sampled = n_sample_trajectory_df(tmp_df_b, n_sample_points=n_sample_points)

        tmp_df_a_sampled = tmp_df_a_sampled[state_action_features]
        tmp_df_a_sampled[target] = 1 if row['Preference'] == "Trajectory_A" else 0


        tmp_df_b_sampled = tmp_df_b_sampled[state_action_features]
        tmp_df_b_sampled[target] = 1 if row['Preference'] == "Trajectory_B" else 0

        tmp_df = pd.concat([tmp_df_a_sampled, tmp_df_b_sampled], axis=0)
        preferences_df_formatted = preferences_df_formatted.append(tmp_df)

    preferences_df_formatted = preferences_df_formatted.reset_index(drop=True)
    
    # Training the model and evaluate with k-fold
    X = preferences_df_formatted[state_action_features]
    y = preferences_df_formatted[target]
    model_hidden_config = [64, 64]
    skf = StratifiedKFold(n_splits=3)
    metrics_report_history = {}
    for i, (train_index, test_index) in enumerate(skf.split(X, y.apply(str))):
        train_df = preferences_df_formatted[preferences_df_formatted.index.isin(train_index)]
        test_df = preferences_df_formatted[preferences_df_formatted.index.isin(test_index)]
        # Training
        features, decisions = train_df[state_action_features].to_numpy(), train_df[target].to_numpy()
        reward_model = MLP(name=reward_model_name, layer_dims=[features.shape[1]+1] + model_hidden_config + [1], out_act=None)
        losses = train_reward_model(
            reward_model,
            features,
            decisions,
            loss_function=preference_loss_function,
            learning_rate=0.0001,
            num_epochs=num_epochs,
            batch_size=256,
            save_dir=save_dir)

        # K-Fold testing
        test_features, test_decisions = test_df[state_action_features].to_numpy(), test_df[target].to_numpy()
        predictions, metrics_report = evaluate_model(reward_model, test_features, test_decisions)
        metrics_report_history[i] = metrics_report

    # the json file where the output must be stored
    with open(model_eval_path, "w") as f:
        json.dump(metrics_report_history, f)

    # Saving the learning curves without showing
    plt.figure(figsize=(12, 3))
    plt.plot(losses)
    plt.title("Training loss of reward modelling", size=18)
    plt.xlabel("Epoch", size=18)
    plt.ylabel("Loss", size=18)
    plt.savefig(training_curve_path)
    plt.clf()
    
    
if __name__ == "__main__":
    train()
