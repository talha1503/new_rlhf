import os
import pandas as pd
import numpy as np

csv_path = 'D:\Work\EleutherAI\\fairness_gym\ml-fairness-gym\data\metrics\metric_trajectories_fair.csv'
df = pd.read_csv(csv_path)

outdir_path = 'D:\Work\EleutherAI\\fairness_gym\ml-fairness-gym\data\\fair_trajectories'

max_val = max(np.unique(df['sample_id']))
prefix = 'fair_'
for i in range(0, max_val+1):
    subset_df = df[df['sample_id'] == i]
    subset_df.to_csv(os.path.join(outdir_path, prefix + str(i) + '.csv'), index=False)