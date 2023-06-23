import os
import numpy as np
import random
from tqdm.auto import tqdm
import pandas as pd

import openai

openai.api_key = open("../chatgpt/config/key.txt", "r").read().strip("\n")


def get_options_df(trajectories_folder, trajectory_A_name, trajectory_B_name):
    trajectory_A_path = f"{trajectories_folder}/{trajectory_A_name}"
    trajectory_B_path = f"{trajectories_folder}/{trajectory_B_name}"
    trajectory_A_df = pd.read_csv(trajectory_A_path, index_col=[0])
    trajectory_B_df = pd.read_csv(trajectory_B_path, index_col=[0])
    return trajectory_A_df, trajectory_B_df

def get_gpt_response_decision(response):
    decision_txt = f"policy {response.lower().split('policy ')[1].strip()[0]}"
    return decision_txt

def get_gpt_response_reason(response):
    reason_txt = response.lower().split('.')[1].replace('\n', '')
    return reason_txt

def randomly_sampled_comparisons(trajectories_folder, num_samples=20):
    # Get all the available trajectories in the provided trajectory folder
    trajectories = [i for i in os.listdir(trajectories_folder) if '.csv' in i]
    comparison_pairs = {"Trajectory_A": [], "Trajectory_B": []}
    for _ in tqdm(range(int(num_samples*1))):
        # Extract two randomly trajectories
        trajectory_A, trajectory_B = random.sample(trajectories, 2)
        comparison_pairs["Trajectory_A"].append(trajectory_A)
        comparison_pairs["Trajectory_B"].append(trajectory_B)
    comaprison_pairs_df = pd.DataFrame(comparison_pairs)
    comaprison_pairs_df = comaprison_pairs_df.drop_duplicates()
    comaprison_pairs_df = comaprison_pairs_df.head(num_samples)
    return comaprison_pairs_df
    
def extract_bulk_chatgpt_preferences(comaprison_pairs_df, decision_definition, trajectories_folder, n_sample_points=50, n_decimal_places=3, max_decisions=1000, policy_group_order=False, reverse_check=False, model_code="gpt-3.5-turbo"):
    res_df = comaprison_pairs_df.reset_index(drop=True).head(max_decisions)
    # Extract preferences randomly
    generated_decisions_list = []
    generated_decisions_init_list = []
    generated_decisions_list_reversed = []
    for idx, row in tqdm(res_df.iterrows(), total=len(res_df)):
        # Get the data of the sampled trajectories
        trajectory_A_df, trajectory_B_df = get_options_df(trajectories_folder, row["Trajectory_A"], row["Trajectory_B"])
        ## Extract preference
        # response_0, history_0 = get_gpt_synthetic_preferences(trajectory_A_df, trajectory_B_df, n_sample_points, n_decimal_places, decision_definition, policy_group_order=True)
        response_1, history_1 = get_gpt_synthetic_preferences(trajectory_A_df, trajectory_B_df, n_sample_points, n_decimal_places, decision_definition, policy_group_order=policy_group_order)
        # generated_decisions_init_list.append(response_0.split("AI: ")[1])
        generated_decisions_list.append(response_1.split("AI: ")[1])
        if reverse_check:
            response_2, history_2 = get_gpt_synthetic_preferences(trajectory_B_df, trajectory_A_df, n_sample_points, n_decimal_places, decision_definition)
            generated_decisions_list_reversed.append(response_2.split("AI: ")[1])
        if idx >= max_decisions:
            break
        
    # Convert the data type to pd.DataFrame
    # res_df["preference_init"] = generated_decisions_init_list
    res_df["response"] = generated_decisions_list
    # Parse the response into preference and reasoning before returning the output
    res_df['preference'] = res_df['response'].apply(lambda x: get_gpt_response_decision(x))
    res_df['reason'] = res_df['response'].apply(lambda x: get_gpt_response_reason(x))
    
    if reverse_check:
        res_df["response_reversed"] = generated_decisions_list_reversed
        res_df['preference_reversed'] = res_df['response_reversed'].apply(lambda x: get_gpt_response_decision(x))
        res_df['invariance'] = res_df.apply(lambda x: x["preference"] == x["preference_reversed"], axis=1)
        
    return res_df

########################################################################
# ChatGPT promting response
########################################################################

def init_prompts(txt):
    history = []
    prompt_ = f"Human: {txt} AI: I would recommend policy "
    return history, prompt_
    
def generate_gpt_response(model_code, prompt_, history):
    """ Because gpt-3.5-turbo performs at a similar capability to text-davinci-003 
        but at 10% the price per token, 
        we recommend gpt-3.5-turbo for most use cases.
    """
    if model_code == "gpt-4":
        response = openai.ChatCompletion.create(
                  model="gpt-4", # this is costing $0.002 per 1k tokens
                  messages=[{"role": "user", 
                             "content": prompt_}]
                )
        prompt_ += response.choices[0].message.content
        history.append((prompt_, response.choices[0].message.content.strip()))
    
    elif model_code == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", # this is costing $0.03 per 1k tokens
          messages=[{"role": "user", 
                     "content": prompt}]
        )
        prompt_ += response.choices[0].message.content
        history.append((prompt_, response.choices[0].message.content.strip()))

    else:
        response = openai.Completion.create(
                model="text-davinci-003", # this is costing $0.03 per 1k tokens
                prompt=prompt,
                temperature=0.9,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=["Human:", " AI: I would recommend policy"],
            )

        prompt_ += response.choices[0].text.strip()
        history.append((prompt_, response.choices[0].text.strip()))

    # Save output
    prompt_output_path = f"prompts-{model_code}.txt"
    with open(prompt_output_path, "a") as f:
        f.write(prompt_)
    
    return prompt_, history

def parse_gpt_preference(gpt_response_list):
    response_filtered = [i.split('\nAI')[-1] for i in gpt_response_list]
    return [f"option {' '.join(i.split('policy ')[1]).split(' ')[0]}" for i in response_filtered]

def sample_trajectories(trajectory_A_df, trajectory_B_df, n_sample_points=50):
    sample_step_size = len(trajectory_A_df) // n_sample_points
    trajectory_A_df = trajectory_A_df.reset_index(drop=True)
    trajectory_B_df = trajectory_B_df.reset_index(drop=True)
    sample_index = [i for i in range(len(trajectory_A_df)) if i%sample_step_size == 0]
    trajectory_A_df_sampled = trajectory_A_df[trajectory_A_df.index.isin(sample_index)]
    trajectory_B_df_sampled = trajectory_B_df[trajectory_B_df.index.isin(sample_index)]
    return trajectory_A_df_sampled, trajectory_B_df_sampled

def get_gpt_preference_wrapper(A_acceptance_rate_1=[0.99, 0.98, 0.97, 0.95],
                               A_acceptance_rate_2=[0.99, 0.98, 0.97, 0.95],
                               A_default_rate_1=[0.38, 0.38, 0.37, 0.38],
                               A_default_rate_2=[0.38, 0.38, 0.37, 0.38],
                               A_avg_credit_1=[3.6, 2.6, 3.6, 3.4],
                               A_avg_credit_2=[3.6, 2.6, 3.6, 3.4],
                               B_acceptance_rate_1=[1, 1, 1, 0.99],
                               B_acceptance_rate_2=[1, 1, 1, 0.99],
                               B_default_rate_1=[0.49, 0.50, 0.49, 0.50],
                               B_default_rate_2=[0.49, 0.50, 0.49, 0.50],
                               B_avg_credit_1=[2.6, 3.6, 2.6, 3.6],
                               B_avg_credit_2=[2.6, 3.6, 2.6, 3.6],
                               decimal_place = 2,
                               decision_definition = "Ensure the profit is maximise.",
                               model_code = "gpt-3.5-turbo",
                               policy_group_order=False
                               ):
    
    # For policy A
    A_acceptance_rate_str_1 = ", ".join([str(round(i, decimal_place)) for i in A_acceptance_rate_1])
    A_acceptance_rate_str_2 = ", ".join([str(round(i, decimal_place)) for i in A_acceptance_rate_2])
    A_default_rate_str_1 = ", ".join([str(round(i, decimal_place)) for i in A_default_rate_1])
    A_default_rate_str_2 = ", ".join([str(round(i, decimal_place)) for i in A_default_rate_2])
    A_avg_credit_str_1 = ", ".join([str(round(i, decimal_place)) for i in A_avg_credit_1])
    A_avg_credit_str_2 = ", ".join([str(round(i, decimal_place)) for i in A_avg_credit_2])
    # For policy B
    B_acceptance_rate_str_1 = ", ".join([str(round(i, decimal_place)) for i in B_acceptance_rate_1])
    B_acceptance_rate_str_2 = ", ".join([str(round(i, decimal_place)) for i in B_acceptance_rate_2])
    B_default_rate_str_1 = ", ".join([str(round(i, decimal_place)) for i in B_default_rate_1])
    B_default_rate_str_2 = ", ".join([str(round(i, decimal_place)) for i in B_default_rate_2])
    B_avg_credit_str_1 = ", ".join([str(round(i, decimal_place)) for i in B_avg_credit_1])
    B_avg_credit_str_2 = ", ".join([str(round(i, decimal_place)) for i in B_avg_credit_2])

    scenario_description = f"A bank is deciding between two policies for giving out loans."
    
    # Schema 1
    option_A_statistics = f"Under policy A:" + " "
    option_A_statistics += f"Acceptance rate of group 1={A_acceptance_rate_str_1} Acceptance rate of group 2= {A_acceptance_rate_str_2}" + " "
    option_A_statistics += f"Default rate of group 1={A_default_rate_str_1} Default rate of group 2= {A_default_rate_str_2}" + " "
    option_A_statistics += f"Average credit score of group 1={A_avg_credit_str_1} Average credit score of group 2= {A_avg_credit_str_2}" + " "
    
    option_B_statistics = f"Under policy B:" + " "
    option_B_statistics += f"Acceptance rate of group 1={B_acceptance_rate_str_1} Acceptance rate of group 2= {B_acceptance_rate_str_2}" + " "
    option_B_statistics += f"Default rate of group 1={B_default_rate_str_1} Default rate of group 2= {B_default_rate_str_2}" + " "
    option_B_statistics += f"Average credit score of group 1={B_avg_credit_str_1} Average credit score of group 2= {B_avg_credit_str_2}" + " "
    
#     # Schema 2
    group_1_statistics = f"For group 1:" + " "
    group_1_statistics += f"Acceptance rate under policy A={A_acceptance_rate_str_1} and under policy B={B_acceptance_rate_str_1}" + " "
    group_1_statistics += f"Default rate under policy A={A_default_rate_str_1} Default rate under policy B={B_default_rate_str_1}" + " "
    group_1_statistics += f"Average credit score under policy A={A_avg_credit_str_1} Average credit score under policy B={B_avg_credit_str_1}" + " "
    
    group_2_statistics = f"For group 2:" + " "
    group_2_statistics += f"Acceptance rate under policy A={A_acceptance_rate_str_2} and under policy B={B_acceptance_rate_str_2}" + " "
    group_2_statistics += f"Default rate under policy A={A_default_rate_str_2} Default rate under policy B={B_default_rate_str_2}" + " "
    group_2_statistics += f"Average credit score under policy A={A_avg_credit_str_2} Average credit score under policy B={B_avg_credit_str_2}" + " "
    
    question = """Firstly, only give me the policy, which policy is better based on the provided definition? Secondly, start with a new line and strating the word reasoning, provide your reasoning for the decision."""
    
    # Init new gpt model with a pre-defined prompt
    if policy_group_order:
        main_txt = f"""{decision_definition} {scenario_description} {option_A_statistics} {option_B_statistics} {question}"""
    else:
        main_txt = f"""{decision_definition} {scenario_description} {group_1_statistics} {group_2_statistics} {question}"""
    history, prompt_ = init_prompts(main_txt)
    # Extract response
    answer, history = generate_gpt_response(model_code, prompt_, history)
    return answer, history
    
def get_gpt_synthetic_preferences(trajectory_A_df_raw, 
                                  trajectory_B_df_raw,
                                  n_sample_points=50,
                                  decimal_place = 2,
                                  decision_definition = "Ensure the profit is maximise.",
                                  question="",
                                  model_code = "gpt-3.5-turbo",
                                  policy_group_order=False
                                 ):
    trajectory_A_df, trajectory_B_df = sample_trajectories(trajectory_A_df_raw, trajectory_B_df_raw, n_sample_points)
    answer, history = get_gpt_preference_wrapper(
           trajectory_A_df["acceptance_rate-group_1"].to_list(),
           trajectory_A_df["acceptance_rate-group_2"].to_list(),
           trajectory_A_df["default_rate-group_1"].to_list(),
           trajectory_A_df["default_rate-group_2"].to_list(),
           trajectory_A_df["average_credit_score-group_1"].to_list(),
           trajectory_A_df["average_credit_score-group_2"].to_list(),
           trajectory_B_df["acceptance_rate-group_1"].to_list(),
           trajectory_B_df["acceptance_rate-group_2"].to_list(),
           trajectory_B_df["default_rate-group_1"].to_list(),
           trajectory_B_df["default_rate-group_2"].to_list(),
           trajectory_B_df["average_credit_score-group_1"].to_list(),
           trajectory_B_df["average_credit_score-group_2"].to_list(),
           decimal_place=decimal_place,
           decision_definition=decision_definition,
           model_code=model_code, 
           policy_group_order=policy_group_order
        )
    return answer, history
