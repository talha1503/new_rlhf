# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.distributions.categorical import Categorical
from rewards import reward_function, update_equalized_group_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    # fmt: off
    class Args():
        def __init__(self):
            pass
    
    args = Args()
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
    #     help="the name of this experiment")
    # parser.add_argument("--seed", type=int, default=1,
    #     help="seed of the experiment")
    # parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, `torch.backends.cudnn.deterministic=False`")
    # parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="if toggled, cuda will be enabled by default")
    # parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="if toggled, this experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to capture videos of the agent performances (check out `videos` folder)")

    # # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="CartPole-v1",
    #     help="the id of the environment")
    # parser.add_argument("--total-timesteps", type=int, default=500000,
    #     help="total timesteps of the experiments")
    # parser.add_argument("--learning-rate", type=float, default=2.5e-4,
    #     help="the learning rate of the optimizer")
    # parser.add_argument("--num-envs", type=int, default=1,
    #     help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=128,
    #     help="the number of steps to run in each environment per policy rollout")
    # parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggle learning rate annealing for policy and value networks")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #     help="the discount factor gamma")
    # parser.add_argument("--gae-lambda", type=float, default=0.95,
    #     help="the lambda for the general advantage estimation")
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="the number of mini-batches")
    # parser.add_argument("--update-epochs", type=int, default=4,
    #     help="the K epochs to update the policy")
    # parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles advantages normalization")
    # parser.add_argument("--clip-coef", type=float, default=0.2,
    #     help="the surrogate clipping coefficient")
    # parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #     help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    # parser.add_argument("--ent-coef", type=float, default=0.01,
    #     help="coefficient of the entropy")
    # parser.add_argument("--vf-coef", type=float, default=0.5,
    #     help="coefficient of the value function")
    # parser.add_argument("--max-grad-norm", type=float, default=0.5,
    #     help="the maximum norm for the gradient clipping")
    # parser.add_argument("--target-kl", type=float, default=None,
    #     help="the target KL divergence threshold")
    # args = parser.parse_args()
    args.exp_name = 'test'
    args.seed = 1
    args.torch_deterministic = True
    args.cuda = True
    args.track = False
    args.wandb_project_name = "cleanRL"
    args.wandb_entity = None
    args.capture_video = False
    args.env_id = "CartPole-v1"
    args.total_timesteps = 20000
    args.learning_rate = 2.5e-4
    args.num_envs = 1
    args.num_steps = 500
    args.anneal_lr = True
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.num_minibatches = 4
    args.update_epochs = 4
    args.norm_adv = True
    args.clip_coef = 0.2
    args.clip_vloss = True
    args.ent_coef = 0.01
    args.vf_coef = 0.5
    args.max_grad_norm = 0.5
    args.target_kl = None
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    # check gradients
    # normalize reward function
    # 
    return args

def get_summary_features(env, state):
    group_id = env.state.group_id
    acceptance_rates = env.state.acceptance_rates
    default_rates = env.state.default_rates
    avg_credit_scores = env.state.average_credit_score

    summary_stats = torch.tensor([acceptance_rates[group_id], default_rates[group_id], avg_credit_scores[group_id]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
    # state = torch.cat([state, summary_stats], dim=1)
    return state

# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # obs_space = len(envs.observation_space['applicant_features'].nvec) + 3
        obs_space = len(envs.observation_space['applicant_features'].nvec)
        action_space = 2
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_space), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # print("Action size: ", action.size()) 
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train(envs):
    args = parse_args()
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # obs_space = len(envs.observation_space['applicant_features'].nvec) + 3
    obs_space = len(envs.observation_space['applicant_features'].nvec)
    action_space = 2
    obs = torch.zeros((args.num_steps, args.num_envs) + (obs_space, )).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (1, )).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    num_updates = args.total_timesteps // args.batch_size
    
    for update in tqdm(range(1, num_updates + 1), total=num_updates):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        start_time = time.time()
        next_obs = envs.reset()
        next_done = torch.zeros(args.num_envs).to(device)
        

        prev_bank_cash = next_obs['bank_cash']
        one_hot_applicant_features = next_obs['applicant_features']
        applicant_features_tensor = torch.tensor(one_hot_applicant_features, device=device, dtype=torch.float32).unsqueeze(0)
        applicant_features_summary_stats = get_summary_features(envs, applicant_features_tensor)
        equalized_group_dict = {'tp_0': 0, 'tp_1': 0, 'fn_0': 0, 'fn_1': 0}

    
        temp_reward = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = applicant_features_summary_stats
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # print("Input: ", applicant_features_summary_stats)
                action, logprob, _, value = agent.get_action_and_value(applicant_features_summary_stats)
                # print("Action: ", action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if envs.state.bank_cash <=0:
                break
            next_obs, reward, done, info = envs.step(action.item())
            current_bank_cash = next_obs['bank_cash']
            one_hot_next_applicant_features = next_obs['applicant_features']
            next_state_reshaped = torch.tensor(one_hot_next_applicant_features, device=device, dtype=torch.float32).unsqueeze(0)
            applicant_features_summary_stats = get_summary_features(envs, next_state_reshaped)
            equalized_group_dict = update_equalized_group_dict(equalized_group_dict, envs.state.group_id,
                                                               envs.state.will_default, action)
            reward = reward_function(envs, action, prev_bank_cash, current_bank_cash, equalized_group_dict, use_reward_model=False, acceptance_rates=envs.state.acceptance_rates)
            # print("Reward: ", reward)
            temp_reward.append(reward)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = next_obs, torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    break
            prev_bank_cash = current_bank_cash
            if done:
                break
        # if len(temp_reward) > 0:
        # print("Average reward: ", sum(temp_reward)/len(temp_reward))
            
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(applicant_features_summary_stats).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # obs_space = len(envs.observation_space['applicant_features'].nvec) + 3
        obs_space = len(envs.observation_space['applicant_features'].nvec)
        action_space = 2
        b_obs = obs.reshape((-1,) + (obs_space, ))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (1, ))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # print("B ACTIONS: ", b_actions.size())
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # envs.close()
    # writer.close()