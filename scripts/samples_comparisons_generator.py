import pandas as pd
from tqdm.auto import tqdm
import click

@click.command()
@click.option('--trajectory_path', default='../data/metrics/metric_trajectories_test.csv')
@click.option('--export_folder_path', default='../data/')
@click.option('--num_sampling', default=100)
@click.option('--burnin_period', default=200)
@click.option('--starting_seed_count', default=0)
def survey_comparisons_sampling(trajectory_path, 
                                export_folder_path, 
                                num_sampling=100, 
                                burnin_period=200, 
                                starting_seed_count=0
                               ):
    test_trajectories_df = pd.read_csv(trajectory_path)
    sample_ids = test_trajectories_df['sample_id'].sample(num_sampling).to_list()
    for tmp_id in tqdm(sample_ids):
        # Take one random sample id 
        # Only sample the period after the burnin period
        tmp_df = test_trajectories_df[test_trajectories_df["sample_id"] == tmp_id].iloc[burnin_period:]
        tmp_df.to_csv(f"{export_folder_path}Max-util_seed{starting_seed_count}.csv")
        starting_seed_count += 1


if __name__ == '__main__':
    survey_comparisons_sampling()