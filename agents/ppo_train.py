import os
from datetime import datetime

import torch
import numpy as np

from agents import ppo
from rewards import reward_function, update_equalized_group_dict

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


def train(env, num_steps, include_summary_stats, checkpoint_path, simulation_iterator, max_ep_len, update_timestep, K_epochs, lr_actor, lr_critic, load_checkpoint_path=None, use_reward_model=False):
    has_continuous_action_space = False  # continuous action space; else discrete
    print_freq = 500  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = 25  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    random_seed = 10  # set random seed if required (0 = no random seed)
    #####################################################
    # state space dimension
    state_dim = len(env.observation_space['applicant_features'].nvec)
    if include_summary_stats:
        state_dim += 3
    # action space dimension
    action_dim = 2

    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = ppo.PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std, include_summary_stats)

    if load_checkpoint_path:
        ppo_agent.load(load_checkpoint_path)
    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= num_steps:
        state = env.reset()
        current_ep_reward = 0
        prev_bank_cash = state['bank_cash']
        one_hot_applicant_features = state['applicant_features']
        applicant_features_tensor = torch.tensor(one_hot_applicant_features, device=device, dtype=torch.float32).unsqueeze(0)
        if include_summary_stats:
            applicant_features_summary_stats = get_summary_features(env, applicant_features_tensor)
        else:
            applicant_features_summary_stats = applicant_features_tensor

        equalized_group_dict = {'tp_0': 0, 'tp_1': 0, 'fn_0': 0, 'fn_1': 0}
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(applicant_features_summary_stats)
            next_state, reward, done, _ = env.step(action)
            current_bank_cash = next_state['bank_cash']
            one_hot_next_applicant_features = next_state['applicant_features']
            if include_summary_stats:
                next_state_reshaped = torch.tensor(one_hot_next_applicant_features, device=device, dtype=torch.float32).unsqueeze(0)
                next_state_reshaped = get_summary_features(env, next_state_reshaped)

            equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                               env.state.will_default, action)
            reward = reward_function(env, action, prev_bank_cash, current_bank_cash, equalized_group_dict, use_reward_model=use_reward_model, acceptance_rates=env.state.acceptance_rates)

            # Do we save only the reward and not the states
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # Do we update here? or de keep this as the same?
            # one_hot_state = one_hot_next_state
            prev_bank_cash = current_bank_cash

            # update PPO agent
            if time_step % update_timestep == 0 and time_step!=0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0 and time_step!=0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                if print_running_episodes!=0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                            print_avg_reward))
                    print_running_reward = 0
                    print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0 and time_step!=0:
                # print("--------------------------------------------------------------------------------------------")
                ppo_agent.save(checkpoint_path)
                # print("model saved")
                # print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()
