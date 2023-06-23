import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from rewards import reward_function, update_equalized_group_dict

from agents import ppo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_summary_features(env, state):
    group_id = env.state.group_id
    acceptance_rates = env.state.acceptance_rates
    default_rates = env.state.default_rates
    avg_credit_scores = env.state.average_credit_score

    summary_stats = torch.tensor([acceptance_rates[group_id], default_rates[group_id], avg_credit_scores[group_id]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.cat([state, summary_stats], dim=1)
    return state


def test(env, include_summary_stats, model_checkpoint_path, num_steps, max_ep_len):
    has_continuous_action_space = False
    action_std = 0.1

    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001

    # state space dimension
    state_dim = len(env.observation_space['applicant_features'].nvec)
    if include_summary_stats:
        state_dim += 3
    # action space dimension
    action_dim = 2

    # initialize a PPO agent
    ppo_agent = ppo.PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std, include_summary_stats)

    random_seed = 10  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    print("loading network from : " + model_checkpoint_path)

    ppo_agent.load(model_checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, num_steps + 1):
        ep_reward = 0
        state = env.reset()
        prev_bank_cash = state['bank_cash']
        one_hot_applicant_features = state['applicant_features']
        applicant_features_tensor = torch.tensor(one_hot_applicant_features, device=device,
                                                 dtype=torch.float32).unsqueeze(0)
        if include_summary_stats:
            applicant_features_summary_stats = get_summary_features(env, applicant_features_tensor)
        else:
            applicant_features_summary_stats = applicant_features_tensor

        equalized_group_dict = {'tp_0': 0, 'tp_1': 0, 'fn_0': 0, 'fn_1': 0}
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(applicant_features_summary_stats)
            next_state, reward, done, _ = env.step(action)
            current_bank_cash = next_state['bank_cash']
            one_hot_next_applicant_features = next_state['applicant_features']
            if include_summary_stats:
                next_state_reshaped = torch.tensor(one_hot_next_applicant_features, device=device,
                                                   dtype=torch.float32).unsqueeze(0)
                next_state_reshaped = get_summary_features(env, next_state_reshaped)

            equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                               env.state.will_default, action)
            reward = reward_function(env, action, prev_bank_cash, current_bank_cash, equalized_group_dict)
            ep_reward += reward
            prev_bank_cash = current_bank_cash

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    avg_test_reward = test_running_reward / num_steps
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")
