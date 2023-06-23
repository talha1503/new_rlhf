import os
import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
import sys
sys.path.append("../")
from source.rule_based_selectors import get_greedy_synthetic_preferences, get_fair_synthetic_preferences
from source.gpt_selector import (get_options_df,
                                 parse_gpt_response,
                                 randomly_sampled_comparisons,
                                 extract_bulk_chatgpt_preferences,
                                 get_gpt_synthetic_preferences, 
                                 parse_gpt_preference,
                                )
import clicks

@click.command()
@click.option("--comparison_options_path", default="./data/comparison_options.csv")
@click.option("--n_decisions", default=20)
@click.option("--n_sample_points", default=50)
@click.option("--stats_decimals", default=5)
@click.option("--mode", default="fair")
@click.option("--model_code", default="gpt-3.5-turbo")
@click.option("--output_folder", default="./data")
def generate_chatgpt_preferences(comparison_options_path, n_sample_points, mode, model_code, output_folder):
    if mode == "fair":
        decision_definition = "Equal opportunity fairness which ensure the same percentage of individuals who are as likely to succeed at loans are given loans."    
    else:
        decision_definition = "Ensure the profit is maximise."
        
    # Load the comparison dataframe
    comaprison_pairs_df = pd.read_csv(comparison_options_path)
    # Get synthetic preferences
    gpt_response_df = extract_bulk_chatgpt_preferences(comaprison_pairs_df, 
                                                       decision_definition,
                                                       n_sample_points,
                                                       n_decimal_places=3,
                                                       max_decisions=1000,
                                                       policy_group_order=False,
                                                       reverse_check=False
                                                      )
    # Export
    gpt_response_df.to_csv(f"{output_folder}/synthetic_decisions.csv", index=False)

if __name__ == "__main__":
    generate_chatgpt_preferences()
