import numpy as np
import random
from rewards import reward_function, update_equalized_group_dict


def train_qlearning(env, num_steps, simulation_iterator):
    q_table = np.zeros([len(env.observation_space['applicant_features'].nvec), 2])
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    for _ in simulation_iterator(num_steps):
        state = env.reset()  # Reject: 0, Accept: 1
        prev_bank_cash = state['bank_cash']
        applicant_features = np.argmax(state['applicant_features'])
        equalized_group_dict = {'tp_0': 0, 'tp_1': 0, 'fn_0': 0, 'fn_1': 0}
        for i in range(100):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[applicant_features])

            next_state, reward, done, _ = env.step(action)
            current_bank_cash = next_state['bank_cash']
            next_applicant_features = np.argmax(next_state['applicant_features'])

            equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                               env.state.will_default, action)

            reward = reward_function(env, action, prev_bank_cash, current_bank_cash,
                                                             equalized_group_dict)

            old_value = q_table[applicant_features, action]
            next_max = np.max(q_table[next_applicant_features])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[applicant_features, action] = new_value

            prev_bank_cash = current_bank_cash
            applicant_features = next_applicant_features

            if done:
                break
