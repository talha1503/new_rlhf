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

"""Metrics for lending environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import core
from environments import lending
import utils
import numpy as np
from typing import Dict, Text
import pandas as pd

class CreditDistribution(core.Metric):
  """Metric that returns a report of the population credit distribution."""

  def __init__(self, env, step=-1, realign_fn=None):
    super(CreditDistribution, self).__init__(env, realign_fn=realign_fn)
    self.step = step

  def measure(self, env):
    """Returns the distribution of credit scores for each group."""
    history = self._extract_history(env)
    params = history[self.step].state.params
    result = {}
    for component in params.applicant_distribution.components:
      group_id = np.argmax(component.components[0].group_membership.mean)
      result[str(group_id)] = component.weights
    return result


class CumulativeLoans(core.Metric):
  """Returns the cumulative number of loans given to each group over time."""

  def measure(self, env):
    """Returns an array of size (num_groups) x (num_steps).

    Cell (i, j) contains the cumulative number of loans given at time j to group
    members of group i.

    Args:
      env: The environment to be measured.
    """

    history = self._extract_history(env)
    result = []
    for history_item in history:
      state = history_item.state  # type: lending.State  # pytype: disable=annotation-type-mismatch
      # Take advantage of the one-hot encoding of state.group in order to build
      # a (num_steps) x (num_groups) array with 1s where loans were given.
      # Multiplying by action makes a row of all zeros if the loan was rejected.
      result.append(np.array(state.group) * history_item.action)
    return np.cumsum(result, 0).T


class CumulativeRecall(core.Metric):
  """Returns the recall aggregated up to time T."""

  def measure(self, env):
    """Returns an array of size (num_groups) x (num_steps).

    Cell (i, j) contains the recall up to time j for group i.

    Args:
      env: The environment to be measured.
    """

    history = self._extract_history(env)
    numerator = []
    denominator = []
    for history_item in history:
      state = history_item.state  # type: lending.State  # pytype: disable=annotation-type-mismatch
      numerator.append(
          np.array(state.group) * history_item.action *
          (1 - state.will_default))
      denominator.append(np.array(state.group) * (1 - state.will_default))
    return (np.cumsum(numerator, 0) / np.cumsum(denominator, 0)).T

  
class AcceptanceRate(core.Metric):
  def measure(self,env):
    history = self._extract_history(env)
    loan_distriubtion = {0: {"total": 0, "approved": 0 }, 1: {"total": 0, "approved": 0 } }
    acceptance_rates = []
    for history_item in history:
      state = history_item.state

      if history_item.action == 1:
        loan_distriubtion[state.group_id]["approved"] += 1
      loan_distriubtion[state.group_id]["total"] += 1

      if loan_distriubtion[0]["total"] == 0:
        acceptance_rates.append(( 0,loan_distriubtion[1]["approved"]/loan_distriubtion[1]["total"]))

      elif loan_distriubtion[1]["total"] == 0:
        acceptance_rates.append(( loan_distriubtion[0]["approved"]/loan_distriubtion[0]["total"],0))

      else:
        acceptance_rates.append(( loan_distriubtion[0]["approved"]/loan_distriubtion[0]["total"],loan_distriubtion[1]["approved"]/loan_distriubtion[1]["total"]))
    
    return acceptance_rates
  
class CreditScoreDistribution(core.Metric):
  def measure(self,env):
    credit_distribution = []
    history = self._extract_history(env)
    for history_item in history:
      tmp_credit_distribution = []
      state = history_item.state
      for group_id in [0,1]:
        cluster_probability = state.params.applicant_distribution.components[group_id].weights
        tmp_credit_distribution.append(list(cluster_probability))
      
      credit_distribution.append(tmp_credit_distribution)

    return credit_distribution

class AverageCreditScore(core.Metric):
  def measure(self,env):
    average_credit_score = []
    history = self._extract_history(env)
    for history_item in history:
      average_credit_score_at_t = []
      state = history_item.state
      for group_id in [0,1]:
        cluster_probability = state.params.applicant_distribution.components[group_id].weights
        average_credit_score_at_t.append(sum([index * value for index, value in enumerate(cluster_probability)]))
      
      average_credit_score.append(average_credit_score_at_t)

    return average_credit_score

class DefaultRate(core.Metric):
  def measure(self,env):
    history = self._extract_history(env)
    defaulted_loans = [0,0]
    total_loans = [0,0]
    default_rates = []
    for history_item in history:
      state = history_item.state
      if history_item.action == 0:
        if default_rates:
          cur_defater_rates = default_rates[-1]
        else:
          cur_defater_rates = [0,0]
        default_rates.append(cur_defater_rates)
        continue
      if state.will_default:
        defaulted_loans[state.group_id] += 1
      total_loans[state.group_id] += 1

      if total_loans[0] == 0:
        default_rates.append(( 0,defaulted_loans[1]/total_loans[1]))
      elif total_loans[1]== 0:
        default_rates.append(( defaulted_loans[0]/total_loans[0],0))
      else:
        default_rates.append(( defaulted_loans[0]/total_loans[0],defaulted_loans[1]/total_loans[1]))
    return default_rates