import os
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm


def greedy_synthetic_preferences(trajectory_A_df, trajectory_B_df):
    trajectory_A_df = trajectory_A_df.reset_index(drop=True)
    trajectory_B_df = trajectory_B_df.reset_index(drop=True)

    total_credits_accepted_A = []
    total_credits_accepted_B = []
    
    total_defaulter_accepted_A = []
    total_defaulter_accepted_B = []
    
    for idx, group_i in enumerate(trajectory_A_df["applicant_group_membership"].to_list()):
        total_credits_accepted_A.append(trajectory_A_df.loc[idx, f"average_credit_score-group_{int(group_i)+1}"])
        total_defaulter_accepted_A.append(trajectory_A_df.loc[idx, f"defaulter_rate-group_{int(group_i)+1}"])
        
    for idx, group_i in enumerate(trajectory_B_df["applicant_group_membership"].to_list()):
        total_credits_accepted_B.append(trajectory_B_df.loc[idx, f"average_credit_score-group_{int(group_i)+1}"])
        total_defaulter_accepted_B.append(trajectory_B_df.loc[idx, f"defaulter_rate-group_{int(group_i)+1}"])
    
    sum_credits_accepted_A = sum(total_credits_accepted_A)
    sum_credits_accepted_B = sum(total_credits_accepted_B)
    sum_defaulter_accepted_A = sum(total_defaulter_accepted_A)
    sum_defaulter_accepted_B = sum(total_defaulter_accepted_B)
    
    if sum_credits_accepted_A != sum_credits_accepted_B:
        return "Trajectory_A" if sum_credits_accepted_A > sum_credits_accepted_B else "Trajectory_B"
    elif sum_defaulter_accepted_A != sum_defaulter_accepted_B:
        return "Trajectory_A" if sum_defaulter_accepted_A < sum_defaulter_accepted_B else "Trajectory_B"
    else:
#         print("Random selection used.")
#         random.sample(["Trajectory_A", "Trajectory_B"], 1)[0]
        return None
        
    
def fair_synthetic_preferences(trajectory_A_df, trajectory_B_df):
    trajectory_A_df = trajectory_A_df.reset_index(drop=True)
    trajectory_B_df = trajectory_B_df.reset_index(drop=True)

    total_credits_accepted_A_group_1 = []
    total_credits_accepted_A_group_2 = []
    total_credits_accepted_B_group_1 = []
    total_credits_accepted_B_group_2 = []
    
    for idx in range(len(trajectory_A_df["applicant_group_membership"])):
        total_credits_accepted_A_group_1.append(trajectory_A_df.loc[idx, f"acceptance_rate-group_1"])
        total_credits_accepted_A_group_2.append(trajectory_A_df.loc[idx, f"acceptance_rate-group_2"])
        
    for idx in range(len(trajectory_B_df["applicant_group_membership"])):
        total_credits_accepted_B_group_1.append(trajectory_B_df.loc[idx, f"acceptance_rate-group_1"])
        total_credits_accepted_B_group_2.append(trajectory_B_df.loc[idx, f"acceptance_rate-group_2"])
        
    A_diff = np.array(total_credits_accepted_A_group_1) - np.array(total_credits_accepted_A_group_2)
    B_diff = np.array(total_credits_accepted_B_group_1) - np.array(total_credits_accepted_B_group_2)

    A_n = len(A_diff)
    B_n = len(B_diff)
    
    A_diff_final = np.mean(A_diff[-A_n//4:])
    B_diff_final = np.mean(B_diff[-B_n//4:])
    A_diff_mid = np.mean(A_diff[-3*A_n//4:-A_n//4])
    B_diff_mid = np.mean(B_diff[-3*B_n//4:-B_n//4])
    A_diff_early = np.mean(A_diff[:A_n//4])
    B_diff_early = np.mean(B_diff[:B_n//4])

    if A_diff_final != B_diff_final:
        return "Trajectory_A" if A_diff_final < B_diff_final else "Trajectory_B"
    
    elif A_diff_mid != B_diff_mid:
        return "Trajectory_A" if A_diff_mid < B_diff_final else "Trajectory_B"
    elif A_diff_early != B_diff_early:
        return "Trajectory_A" if A_diff_early < B_diff_early else "Trajectory_B"
    else:
#         random.sample(["Trajectory_A", "Trajectory_B"], 1)[0]
        return None
    
    

n_synthetic_decisions = 1000

trajectories_folder = "data/trajectories/"
trajectories = [i for i in os.listdir(trajectories_folder) if '.csv' in i]

greed_decisions_all = []
fair_decisions_all = []
decision_missed_rate = 0
for _ in tqdm(range(n_synthetic_decisions)):
    trajectory_A, trajectory_B = random.sample(trajectories, 2)

    trajectory_A_df = pd.read_csv(f"data/trajectories/{trajectory_A}", index_col=[0])
    trajectory_B_df = pd.read_csv(f"data/trajectories/{trajectory_B}", index_col=[0])

    greedy_decisions = greedy_synthetic_preferences(trajectory_A_df, trajectory_B_df)
    fair_decisions = fair_synthetic_preferences(trajectory_A_df, trajectory_B_df)
    
    if greedy_decisions != None and fair_decisions != None:
        greed_decisions_all.append([trajectory_A, trajectory_B, greedy_decisions])
        fair_decisions_all.append([trajectory_A, trajectory_B, fair_decisions])
    else:
        decision_missed_rate += 1
    
decision_missed_rate /= n_synthetic_decisions
print(decision_missed_rate)


greedy_decisions_df = pd.DataFrame(greed_decisions_all, columns=["Trajectory_A", "Trajectory_B", "Preference"])
fair_decisions_df = pd.DataFrame(fair_decisions_all, columns=["Trajectory_A", "Trajectory_B", "Preference"])

greedy_decisions_df.to_csv("data/greedy_decisions.csv", index=False)
fair_decisions_df.to_csv("data/fair_decisions.csv", index=False)
