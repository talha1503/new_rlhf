import pandas as pd
from tqdm.auto import tqdm

def survey_comparisons_sampling(trajectory_path, 
                                export_folder_path, 
                                num_sampling=100, , 
                                burnin_period = 200, 
                                starting_seed_count = 0
                               ):
    test_trajectories_df = pd.read_csv(trajectory_sampling_path)
    sample_ids = test_trajectories_df['sample_id'].sample(num_sampling).to_list()
    for tmp_id in tqdm(sample_ids):
        # Take one random sample id 
        # Only sample the period after the burnin period
        tmp_df = test_trajectories_df[test_trajectories_df["sample_id"] == tmp_id].iloc[burnin_period:]
        tmp_df.to_csv(f"{export_folder_path}Max-util_seed{starting_seed_count}.csv")
        starting_seed_count += 1


if __name__ == '__main__':
    survey_comparisons_sampling(trajectory_path="../data/metrics/metric_trajectories_old.csv",
                                export_folder_path=f"data/trajectories/",
                                num_sampling=100, , 
                                burnin_period = 200, 
                                starting_seed_count = 0
                               )
    