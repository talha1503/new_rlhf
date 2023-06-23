import pandas as pd
import numpy as np
import openai
## OpenAI - load and set our key
# openai.api_key = open("../chatgpt/config/key.txt", "r").read().strip("\n")

def get_greedy_synthetic_preferences(trajectory_A_df, trajectory_B_df):
    trajectory_A_df = trajectory_A_df.reset_index(drop=True)
    trajectory_B_df = trajectory_B_df.reset_index(drop=True)

    total_credits_A = []
    total_credits_B = []
    
    total_default_A = []
    total_default_B = []
    
    for idx, group_i in enumerate(trajectory_A_df["applicant_group_membership"].to_list()):
        total_credits_A.append(trajectory_A_df.loc[idx, f"average_credit_score-group_{int(group_i)+1}"])
        total_default_A.append(trajectory_A_df.loc[idx, f"default_rate-group_{int(group_i)+1}"])
        
    for idx, group_i in enumerate(trajectory_B_df["applicant_group_membership"].to_list()):
        total_credits_B.append(trajectory_B_df.loc[idx, f"average_credit_score-group_{int(group_i)+1}"])
        total_default_B.append(trajectory_B_df.loc[idx, f"default_rate-group_{int(group_i)+1}"])
    
    sum_credits_accepted_A = sum(total_credits_A)
    sum_credits_accepted_B = sum(total_credits_B)
    sum_default_accepted_A = sum(total_default_A)
    sum_default_accepted_B = sum(total_default_B)
    
    if sum_credits_accepted_A != sum_credits_accepted_B:
        return "Trajectory_A" if sum_credits_accepted_A > sum_credits_accepted_B else "Trajectory_B"
    elif sum_default_accepted_A != sum_default_accepted_B:
        return "Trajectory_A" if sum_default_accepted_A < sum_default_accepted_B else "Trajectory_B"
    else:
        return None
        
def get_fair_synthetic_preferences(trajectory_A_df, trajectory_B_df):
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
        return None
