import pandas as pd
from tqdm.auto import tqdm
import click

# trajectory_samples_path = "../data/metrics/metric_trajectories_test.csv"
# export_folder_path = f"../data/trajectories/"

num_segments_sampling = 50
burnin_period = 200


@click.command()
@click.option("--trajectory_samples_path", default="./data/metrics/metric_trajectories_test.csv")
@click.option("--export_folder_path", default="./data/trajectories/")
@click.option("--num_segments_sampling", default=50)
@click.option("--burnin_period", default=200)
@click.option("--segment_length", default=100)
def extract_trajectory_segments(trajectory_samples_path,
                                export_folder_path,
                                num_segments_sampling,
                                burnin_period,
                                segment_length
                                ):
    test_trajectories_df = pd.read_csv(trajectory_samples_path)
    sample_ids = test_trajectories_df['sample_id'].sample(num_segments_sampling).to_list()

    starting_seed_count = 0
    for tmp_id in tqdm(sample_ids):
        # Take one random sample id 
        # Only sample the period after the burnin period
        tmp_df = test_trajectories_df[test_trajectories_df["sample_id"] == tmp_id].iloc[burnin_period:]

        if segment_length < len(tmp_df):
            tmp_df = tmp_df.tail(segment_length)

        tmp_df.to_csv(f"{export_folder_path}Max-util_seed{starting_seed_count}.csv")
        starting_seed_count += 1


if __name__ == "__main__":
    extract_trajectory_segments()
