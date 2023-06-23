import os
import numpy as np
import pandas as pd
import random
import click


@click.command()
@click.option("--n_comparisons", default=1000)
@click.option("--trajectories_folder", default="./data/trajectories")
def extract_comparison_samples(n_comparisons, trajectories_folder):
    trajectories = [i for i in os.listdir(trajectories_folder) if '.csv' in i]    
    count = 0
    comaprison_list = []
    while count < n_comparisons:
        # Randomly select two trajectories for comparison
        trajectory_A, trajectory_B = random.sample(trajectories, 2)

        if (trajectory_A != trajectory_B) and [trajectory_A, trajectory_B] not in comaprison_list:
            comaprison_list.append([trajectory_A, trajectory_B])
            count += 1

    comparisons_df = pd.DataFrame(comaprison_list, columns=["Trajectory_A", "Trajectory_B"])
    print("Comparison df shape: ", comparisons_df.shape)
    comparisons_df.to_csv("../data/comparison_options.csv", index=False)

if __name__ == "__main__":
    extract_comparison_samples()
    