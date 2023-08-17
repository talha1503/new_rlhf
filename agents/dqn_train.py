import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from rewards import reward_function, update_equalized_group_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_summary_features(env):
    group_id = env.state.group_id
    acceptance_rates = env.state.acceptance_rates
    default_rates = env.state.default_rates
    avg_credit_scores = env.state.average_credit_score

    summary_stats = torch.tensor([acceptance_rates[group_id], default_rates[group_id], avg_credit_scores[group_id]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
    return summary_stats


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, include_summary_stats):
        self.x = x
        self.y = y
        self.include_summary_stats = include_summary_stats

    def __getitem__(self, index):
        if self.include_summary_stats:
            x1 = torch.tensor(self.x[index][0], device=device, dtype=torch.float32)
            x2 = torch.tensor(self.x[index][1], device=device, dtype=torch.float32)
            x = (x1, x2)
        else:
            x = torch.tensor(self.x[index][0], device=device, dtype=torch.float32)
        y = torch.tensor(self.y[index], device=device, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.x)


class DQNAgent(nn.Module):
    # Try out xavier initialization
    def __init__(self, state_space, action_space, include_summary_stats):
        super().__init__()
        self.include_summary_stats = include_summary_stats
        self.applicant_layer1 = nn.Linear(state_space, 24)
        self.applicant_layer2 = nn.Linear(24, 12)
        if self.include_summary_stats:
            self.summary_layer_1 = nn.Linear(3, 12)
            self.merge_layer = nn.Linear(24, 12)

        self.out_layer = nn.Linear(12, action_space)

    def forward(self, x, summary_stat_features=None):
        applicant_features = F.relu(self.applicant_layer1(x))
        applicant_features = F.relu(self.applicant_layer2(applicant_features))

        if self.include_summary_stats:
            summary_stat_features = self.summary_layer_1(summary_stat_features)
            applicant_features = torch.cat([applicant_features, summary_stat_features], dim=1)
            applicant_features = self.merge_layer(applicant_features)

        x = self.out_layer(applicant_features)
        return x


def train_network(model, x, y, batch_size):
    model = model.to(device)
    dataset = ListDataset(x, y, model.include_summary_stats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    epochs = 100
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0
        for i, (x, y) in enumerate(dataloader, 0):
            optimizer.zero_grad()
            y = y.to(device)
            if model.include_summary_stats:
                x[0] = x[0].to(device)
                x[1] = x[1].to(device)
                x[1] = x[1].squeeze(1)
                outputs = model(x[0], x[1])
            else:
                x = x.to(device)
                outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = (epoch_loss / (i + 1))
        # if epoch % 10 == 0:
        #     # epoch_loss = (epoch_loss/(i+1))
        #     print("Epoch: {} Training Loss: {} Training Accuracy: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model


def train_using_bellman_eq(env, replay_memory, model, target_model, done):
    learning_rate = 0.7  # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return model

    batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)
    current_applicant_states = torch.tensor([transition[0] for transition in mini_batch], device=device,
                                            dtype=torch.float32)
    if model.include_summary_stats:
        current_summary_stats = torch.cat([transition[5] for transition in mini_batch], dim=0)
    else:
        current_summary_stats = None
    current_qs_list = model(current_applicant_states, current_summary_stats)

    new_applicant_states = torch.tensor([transition[3] for transition in mini_batch], device=device,
                                        dtype=torch.float32)
    if model.include_summary_stats:
        new_summary_stats = torch.cat([transition[6] for transition in mini_batch], dim=0)
    else:
        new_summary_stats = None
    future_qs_list = target_model(new_applicant_states, new_summary_stats)

    x, y = [], []
    for index, (state, action, reward, next_state, done, summary_stats, next_summary_stats) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * torch.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q
        x.append((state, summary_stats))
        y.append(current_qs)

    model = train_network(model, x, y, batch_size)
    return model


def train_dqn(env, simulation_iterator, include_summary_stats, num_steps):
    epsilon = 0.1
    model = DQNAgent(len(env.observation_space['applicant_features'].nvec), 2, include_summary_stats)
    target_model = DQNAgent(len(env.observation_space['applicant_features'].nvec), 2, include_summary_stats)

    target_model.load_state_dict(model.state_dict())
    replay_memory = deque(maxlen=50_000)

    steps_to_update_target_model = 0

    model = model.to(device)
    target_model = target_model.to(device)

    for _ in simulation_iterator(num_steps):
        state = env.reset() # Reject: 0, Accept: 1
        prev_bank_cash = state['bank_cash']
        one_hot_applicant_features = state['applicant_features']
        applicant_features = np.argmax(state['applicant_features'])
        equalized_group_dict = {'tp_0': 0, 'tp_1': 0, 'fn_0': 0, 'fn_1': 0}
        done = False
        summary_features = None

        for i in range(100):
            steps_to_update_target_model += 1
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                applicant_features_tensor = torch.tensor(one_hot_applicant_features, device=device, dtype=torch.float32).unsqueeze(0)
                if include_summary_stats:
                    summary_features = get_summary_features(env)
                else:
                    summary_features = None

                prediction = model(applicant_features_tensor, summary_features).squeeze(0)
                action = torch.argmax(prediction).item()

            next_state, _, done, _ = env.step(action)
            current_bank_cash = next_state['bank_cash']
            one_hot_next_applicant_features = next_state['applicant_features']
            next_applicant_features = np.argmax(next_state['applicant_features'])

            if include_summary_stats:
                next_summary_features = get_summary_features(env)
            else:
                next_summary_features = None

            equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                               env.state.will_default, action)

            reward = reward_function(env, action, prev_bank_cash, current_bank_cash,
                                                             equalized_group_dict, acceptance_rates=env.state.acceptance_rates)


            replay_memory.append(
                [one_hot_applicant_features, action, reward, one_hot_next_applicant_features, done, summary_features, next_summary_features])
            ####
            if steps_to_update_target_model % 4 == 0:
                model = train_using_bellman_eq(env, replay_memory, model, target_model, done)

            one_hot_applicant_features = one_hot_next_applicant_features
            prev_bank_cash = current_bank_cash
            state = next_state

            if done:
                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break