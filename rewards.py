# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Reward functions for ML fairness gym.

These transforms are used to extract scalar rewards from state variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional
import core
import numpy as np
import torch


def get_summary_features(env, state):
    group_id = env.state.group_id
    acceptance_rates = env.state.acceptance_rates
    default_rates = env.state.default_rates
    avg_credit_scores = env.state.average_credit_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_stats = torch.tensor([acceptance_rates[group_id], default_rates[group_id], avg_credit_scores[group_id]],
                                 dtype=torch.float32, device=device).unsqueeze(0)
    state = torch.cat([state, summary_stats], dim=1)
    return state

def get_model_input_features(env, action, applicant_credit_group, applicant_group_membership):
    acceptance_rates = env.state.acceptance_rates
    default_rates = env.state.default_rates
    avg_credit_scores = env.state.average_credit_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_input_tensor = torch.tensor([default_rates[0], default_rates[1],
                                        acceptance_rates[0], acceptance_rates[1], 
                                        avg_credit_scores[0], avg_credit_scores[1],
                                        applicant_credit_group, applicant_group_membership, 
                                        action],
                                        dtype=torch.float32, device=device).unsqueeze(0)
    return model_input_tensor


class NullReward(core.RewardFn):
    """Reward is always 0."""

    # TODO(): Find a better type for observations than Any.
    def __call__(self, observation):
        del observation  # Unused.
        return 0


class ScalarDeltaReward(core.RewardFn):
    """Extracts a scalar reward from the change in a scalar state variable."""

    def __init__(self, dict_key, baseline=0):
        """Initializes ScalarDeltaReward.

    Args:
      dict_key: String key for the observation used to compute the reward.
      baseline: value to consider baseline when first computing reward delta.
    """
        self.dict_key = dict_key
        self.last_val = float(baseline)

    # TODO(): Find a better type for observations than Any.
    def __call__(self, observation):
        """Computes a scalar reward from observation.

    The scalar reward is computed from the change in a scalar observed variable.

    Args:
      observation: A dict containing observations.
    Returns:
      scalar reward.
    Raises:
      TypeError if the observed variable indicated with self.dict_key is not a
        scalar.
    """
        # Validates that the state variable is a scalar with this float() call.
        current_val = float(observation[self.dict_key])
        retval = current_val - self.last_val
        self.last_val = current_val
        return retval


class BinarizedScalarDeltaReward(ScalarDeltaReward):
    """Extracts a binary reward from the sign of the change in a state variable."""

    # TODO(): Find a better type for observations than Any.
    def __call__(self, observation):
        """Computes binary reward from state.

    Args:
      observation: A dict containing observations.
    Returns:
      1 - if the state variable has gone up.
      0 - if the state variable has gone down.
      None - if the state variable has not changed.
    Raises:
      TypeError if the state variable indicated with self.dict_key is not a
        scalar.
    """
        delta = super(BinarizedScalarDeltaReward, self).__call__(observation)
        # Validate that delta is a scalar.
        _ = float(delta)
        if delta == 0:
            return None
        return int(delta > 0)


class VectorSumReward(core.RewardFn):
    """Extracts scalar reward that is the sum of a vector state variable.

  e.g.if state.my_vector = [1, 2, 4, 6], then
  VectorSumReward('my_vector')(state) returns 13.
  """

    def __init__(self, dict_key):
        """Initializes VectorSumReward.

    Args:
      dict_key: String key for the state variable used to compute the reward.
    """
        self.dict_key = dict_key

    # TODO(): Find a better type for observations than Any.
    def __call__(self, observation):
        """Computes scalar sum reward from state.

    Args:
      observation: An observation containing dict_key.
    Returns:
      Scalar sum of the vector observation defined by dict_key.
    Raises:
      ValueError if the dict_key is not in the observation.
    """
        if self.dict_key not in observation:
            raise ValueError("dict_key %s not in observation" % self.dict_key)
        return np.sum(observation[self.dict_key])


def update_equalized_group_dict(equalized_group_dict, group, will_default, action):
    if group == 0:
        if action == 1 and not will_default:
            equalized_group_dict['tp_0'] += 1
        if action == 0 and not will_default:
            equalized_group_dict['fn_0'] += 1
    else:
        if action == 1 and not will_default:
            equalized_group_dict['tp_1'] += 1
        if action == 0 and not will_default:
            equalized_group_dict['fn_1'] += 1
    return equalized_group_dict


# def reward_function(env,
#                     action,
#                     prev_bank_cash,
#                     current_bank_cash,
#                     equalized_group_dict,
#                     bank_cash_multiplier=1,
#                     tpr_diff_multiplier=10,
#                     use_reward_model=False,
#                     reward_model_path="./models/greedy_reward_model.pt"
#                     ):
#     """
#     Description:
#         Input features of the reward model as the following:
#                                         'default_rate-group_1', 'default_rate-group_2',
#                                         'acceptance_rate-group_1', 'acceptance_rate-group_2',
#                                         'average_credit_score-group_2', 'average_credit_score-group_1',
#                                         'applicant_credit_group', 'applicant_group_membership',
#                                         'agent_action'
#
#     Return:
#         A scaler value representing preferences captured by the reward model.
#     """
#     if not use_reward_model:
#         if action == 0:
#             return 1e-2
#
#         if current_bank_cash > prev_bank_cash:
#             bank_cash_reward = 1
#         elif current_bank_cash <= prev_bank_cash:
#             bank_cash_reward = -1
#
#         if equalized_group_dict['tp_0'] + equalized_group_dict['fn_0'] == 0:
#             tpr_0 = 0
#         else:
#             tpr_0 = equalized_group_dict['tp_0'] / (equalized_group_dict['tp_0'] + equalized_group_dict['fn_0'])
#
#         if equalized_group_dict['tp_1'] + equalized_group_dict['fn_1'] == 0:
#             tpr_1 = 0
#         else:
#             tpr_1 = equalized_group_dict['tp_1'] / (equalized_group_dict['tp_1'] + equalized_group_dict['fn_1'])
#
#         tpr_diff = abs(tpr_0 - tpr_1)
#         return (bank_cash_multiplier * bank_cash_reward) + (tpr_diff_multiplier * tpr_diff)
#
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = torch.load(reward_model_path)
#         model = model.to(device)
#         # state = env.reset()
#         applicant_credit_group = np.argmax(env.state.applicant_features)
#         applicant_group_membership = env.state.group_id
#         features = get_model_input_features(env, action, applicant_credit_group, applicant_group_membership)
#
#         with torch.inference_mode():
#             y_pred = model(torch.tensor(features, dtype=torch.float32, device=device))
#
#         # y_pred = predict
#         # ions.detach().cpu() > 0.5
#         return y_pred
def reward_function(env,
                    action,
                    prev_bank_cash,
                    current_bank_cash,
                    equalized_group_dict,
                    bank_cash_multiplier=1,
                    tpr_diff_multiplier=10,
                    use_reward_model=False,
                    reward_model_path="./models/reward_model_fair_rule_based.pt"
                    ):
    """
    Description:
        Input features of the reward model as the following:
                                        'default_rate-group_1', 'default_rate-group_2',
                                        'acceptance_rate-group_1', 'acceptance_rate-group_2',
                                        'average_credit_score-group_2', 'average_credit_score-group_1',
                                        'applicant_credit_group', 'applicant_group_membership',
                                        'agent_action'

    Return:
        A scaler value representing preferences captured by the reward model.
    """
    if not use_reward_model:
        if action == 0:
            return 0

        if current_bank_cash > prev_bank_cash:
            bank_cash_reward = 1
        elif current_bank_cash <= prev_bank_cash:
            bank_cash_reward = -1

        if equalized_group_dict['tp_0'] + equalized_group_dict['fn_0'] == 0:
            tpr_0 = 0
        else:
            tpr_0 = equalized_group_dict['tp_0'] / (equalized_group_dict['tp_0'] + equalized_group_dict['fn_0'])

        if equalized_group_dict['tp_1'] + equalized_group_dict['fn_1'] == 0:
            tpr_1 = 0
        else:
            tpr_1 = equalized_group_dict['tp_1'] / (equalized_group_dict['tp_1'] + equalized_group_dict['fn_1'])

        tpr_diff = abs(tpr_0 - tpr_1)
        # For fair: (bank_cash_multiplier * bank_cash_reward) - (tpr_diff_multiplier * tpr_diff) 10 x tpr_diff_multiplier
        return (bank_cash_multiplier * bank_cash_reward) - (tpr_diff_multiplier * tpr_diff)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(reward_model_path)
        model = model.to(device)
        # state = env.reset()
        applicant_credit_group = np.argmax(env.state.applicant_features)
        applicant_group_membership = env.state.group_id
        features = get_model_input_features(env, action, applicant_credit_group, applicant_group_membership)

        with torch.inference_mode():
            y_pred = model(torch.tensor(features, dtype=torch.float32, device=device))

        # y_pred = predict
        # ions.detach().cpu() > 0.5
        return y_pred
