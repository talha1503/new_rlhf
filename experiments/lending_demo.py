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

"""Main file to run lending experiments for demonstration purposes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append("D:\\Fairness\\rlhf-fairness")
import os
import tkinter
import core
import itertools
from absl import app
from absl import flags
from agents import threshold_policies
from experiments import lending
from experiments import lending_plots
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
import pandas as pd
from tqdm.auto import tqdm

file_suffix = "_test"

# Paths
flags.DEFINE_string('outfile', f"./data/raw/sim_results{file_suffix}.json", 'Path to write out results.')
flags.DEFINE_string('metrics_outfile', f"./data/metrics/metric_trajectories{file_suffix}.csv",
                    'Path to write out metric results.')
flags.DEFINE_string('plots_directory', f"data/plots/equalized_opportunity/", 'Directory to write out plots.')
flags.DEFINE_string('hparams_outfile', f"./data/hparams/hparams_trajectories{file_suffix}.csv",
                    'Path to write out hparams results.')
flags.DEFINE_string('model_checkpoint_path', f"./models/baseline_rl_model_{file_suffix}.pt",
                    'Directory to save baseline models.')
flags.DEFINE_string('load_checkpoint_path', None,
                    'Directory to save baseline models.')
flags.DEFINE_string('rl_agent', None,
                    'Specify the rl agent to be used'
                    )
flags.DEFINE_string('classifier_name', "base_classifier", 'Specify the supervised learning model to be used')

# Hyperparameters
flags.DEFINE_bool('equalize_opportunity', False, 'If true, apply equality of opportunity constraints.')
flags.DEFINE_integer('num_steps', 10000, 'Number of steps to run the simulation.')
flags.DEFINE_float('group_0_prob', 0.5, '')
flags.DEFINE_float('interest_rate', 1.0, '')
flags.DEFINE_integer('bank_starting_cash', 10000, '')
flags.DEFINE_integer('burnin', 200, '')
flags.DEFINE_integer('seed', 200, 'Seed value for random generation')
flags.DEFINE_float('cluster_shift_increment', 0.01, '')
# Sampling
flags.DEFINE_bool('sampling_flag', False, 'If true, then using the following parameter ranges.')
flags.DEFINE_list('policy_options', ["equalize_opportunity"], 'The grid for the policy of the agent.')
flags.DEFINE_list('interest_rate_range', [1.0], 'The grid for the initial interest rate range.')
flags.DEFINE_list('bank_starting_cash_range', [10000, 20000, 30000, 40000, 50000],
                  'The grid for the initial bank starting cash value.')
flags.DEFINE_list('seed_range', [1], '')
flags.DEFINE_bool('include_summary_stats', False, 'If true, include summary statistics in the model')

# PPO
flags.DEFINE_integer('max_ep_len', 1000, 'Maximum episode length.')
flags.DEFINE_integer('update_timestep', 10000, 'interval for updating ppo')
flags.DEFINE_integer('K_epochs', 80, 'K epochs to train the PPO')
flags.DEFINE_float('lr_actor', 0.0003, 'learning rate for actor')
flags.DEFINE_float('lr_critic', 0.001, 'learning rate for critic')
flags.DEFINE_bool('test_mode', False, 'If true, test the model in the environment')
## Reward model
flags.DEFINE_bool('use_reward_model', False, 'If true, use the reward model for training the PPO')

FLAGS = flags.FLAGS

# Control float precision in json encoding.
json.encoder.FLOAT_REPR = lambda o: repr(round(o, 3))

MAXIMIZE_REWARD = threshold_policies.ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY


def format_metrics(result):
    metrics = result['metric_results']
    applicant_credit_group, applicant_group_membership, action = [], [], []
    history = result["environment"]["history"]
    for history_item in history:
        state = history_item.state
        applicant_credit_group.append(np.argmax(state.applicant_features))
        applicant_group_membership.append(state.group_id)
        action.append(history_item.action)

    required_metrics = {
        "acceptance_rate": metrics["acceptance_rate"],
        "default_rate": metrics["default_rate"],
        "credit_score_distribution": metrics["credit_score_distribution"],
        "average_credit_score": metrics["average_credit_score"]
    }

    required_features_reformatted = {f"{k}-group_{i}": [] for i in [1, 2] for k in required_metrics.keys()}
    for metric in required_metrics:
        for timestep in required_metrics[metric]:
            required_features_reformatted[metric + "-group_1"].append(timestep[0])
            required_features_reformatted[metric + "-group_2"].append(timestep[1])

    required_features_reformatted["applicant_group_membership"] = applicant_group_membership
    required_features_reformatted["applicant_credit_group"] = applicant_credit_group
    required_features_reformatted["agent_action"] = action

    df = pd.DataFrame.from_dict(required_features_reformatted, orient='columns')
    df.index += 1
    df.index.name = "Timestep"
    return df


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if not FLAGS.sampling_flag:
        np.random.seed(100)
        result = lending.Experiment(
            group_0_prob=FLAGS.group_0_prob,
            interest_rate=FLAGS.interest_rate,
            bank_starting_cash=FLAGS.bank_starting_cash,
            seed=FLAGS.seed,
            num_steps=FLAGS.num_steps,
            burnin=FLAGS.burnin,
            cluster_shift_increment=FLAGS.cluster_shift_increment,
            include_cumulative_loans=True,
            return_json=False,
            threshold_policy=(EQUALIZE_OPPORTUNITY if FLAGS.equalize_opportunity else
                              MAXIMIZE_REWARD)).run(rl_agent=FLAGS.rl_agent,
                                                    include_summary_stats=FLAGS.include_summary_stats,
                                                    classifier_name=FLAGS.classifier_name,
                                                    model_checkpoint_path=FLAGS.model_checkpoint_path,
                                                    max_ep_len=FLAGS.max_ep_len,
                                                    update_timestep=FLAGS.update_timestep,
                                                    K_epochs=FLAGS.K_epochs,
                                                    lr_actor=FLAGS.lr_actor,
                                                    lr_critic=FLAGS.lr_critic,
                                                    test_mode=FLAGS.test_mode,
                                                    load_checkpoint_path=FLAGS.load_checkpoint_path,
                                                    use_reward_model=FLAGS.use_reward_model
                                                    )

        title = ('Eq. opportunity' if FLAGS.equalize_opportunity else 'Max reward')
        metrics = result['metric_results']

        # Format the results to extract the metrics into a dataframe
        metrics_df = format_metrics(result)
        result_json = core.to_json(result)

        # Standalone figure of initial credit distribution
        fig = plt.figure(figsize=(4, 4))
        lending_plots.plot_credit_distribution(
            metrics['initial_credit_distribution'],
            'Initial',
            path=os.path.join(FLAGS.plots_directory,
                              'initial_credit_distribution.png')
            if FLAGS.plots_directory else None,
            include_median=True,
            figure=fig)

        # Initial and final credit distributions next to each other.
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        lending_plots.plot_credit_distribution(
            metrics['initial_credit_distribution'],
            'Initial',
            path=None,
            include_median=True,
            figure=fig)
        plt.subplot(1, 2, 2)

        lending_plots.plot_credit_distribution(
            metrics['final_credit_distributions'],
            'Final - %s' % title,
            path=os.path.join(FLAGS.plots_directory, 'final_credit_distribution.png')
            if FLAGS.plots_directory else None,
            include_median=True,
            figure=fig)

        fig = plt.figure()
        lending_plots.plot_bars(
            metrics['recall'],
            title='Recall - %s' % title,
            path=os.path.join(FLAGS.plots_directory, 'recall.png')
            if FLAGS.plots_directory else None,
            figure=fig)

        fig = plt.figure()
        lending_plots.plot_bars(
            metrics['precision'],
            title='Precision - %s' % title,
            ylabel='Precision',
            path=os.path.join(FLAGS.plots_directory, 'precision.png')
            if FLAGS.plots_directory else None,
            figure=fig)

        fig = plt.figure()
        lending_plots.plot_cumulative_loans(
            {'demo - %s' % title: metrics['cumulative_loans']},
            path=os.path.join(FLAGS.plots_directory, 'cumulative_loans.png')
            if FLAGS.plots_directory else None,
            figure=fig)

        # print('Profit %s %f' % (title, result['metric_results']['profit rate']))
        plt.show()

    else:
        np.random.seed(100)
        policy_options_grid = FLAGS.policy_options
        interest_rate_grid = [round(i, 3) for i in FLAGS.interest_rate_range]
        bank_starting_cash_grid = [int(i) for i in FLAGS.bank_starting_cash_range]
        seed_grid = [int(i) for i in FLAGS.seed_range]
        hparams_grid = [*itertools.product(policy_options_grid,
                                           interest_rate_grid,
                                           bank_starting_cash_grid,
                                           seed_grid)]

        metrics_df = pd.DataFrame()
        hparams_df = pd.DataFrame()
        result = {}
        print("Number of hparams combinations: ", len(hparams_grid))
        for sample_id, tmp_hparams in tqdm(enumerate(hparams_grid), desc="Running different hparams"):
            equalize_opportunity_flag = True if tmp_hparams[0] == 'equalize_opportunity' else False
            tmp_result = lending.Experiment(
                group_0_prob=FLAGS.group_0_prob,
                interest_rate=tmp_hparams[1],
                bank_starting_cash=tmp_hparams[2],
                seed=tmp_hparams[3],
                num_steps=FLAGS.num_steps,
                burnin=FLAGS.burnin,
                cluster_shift_increment=FLAGS.cluster_shift_increment,
                include_cumulative_loans=True,
                return_json=False,
                threshold_policy=(equalize_opportunity_flag)).run(rl_agent=FLAGS.rl_agent,
                                                                  include_summary_stats=FLAGS.include_summary_stats,
                                                                  classifier_name=FLAGS.classifier_name,
                                                                  model_checkpoint_path=FLAGS.model_checkpoint_path,
                                                                  max_ep_len=FLAGS.max_ep_len,
                                                                  update_timestep=FLAGS.update_timestep,
                                                                  K_epochs=FLAGS.K_epochs,
                                                                  lr_actor=FLAGS.lr_actor,
                                                                  lr_critic=FLAGS.lr_critic,
                                                                  test_mode=FLAGS.test_mode,
                                                                  load_checkpoint_path=FLAGS.load_checkpoint_path,
                                                                  use_reward_model=FLAGS.use_reward_model
                                                                  )

            # print({k: i for k, i in tmp_result['metric_results'].items()})
            tmp_metrics_df = format_metrics(tmp_result)
            # To keep track of the hparams of the trajectories
            tmp_metrics_df.insert(0, "sample_id", sample_id)
            # Log the hyperparameter details
            tmp_hparams_df = pd.DataFrame([tmp_hparams],
                                          columns=["policy", "interest_rate", "bank_starting_cash", "seed"])
            tmp_hparams_df.insert(0, "sample_id", sample_id)
            # Add to the log for export
            metrics_df = metrics_df.append(tmp_metrics_df)
            hparams_df = hparams_df.append(tmp_hparams_df)
            # fig = plt.figure(figsize=(4, 4))
            # metrics = tmp_result['metric_results']
            # title = ('Eq. opportunity' if FLAGS.equalize_opportunity else 'Max reward')
            # lending_plots.plot_credit_distribution(
            #     metrics['initial_credit_distribution'],
            #     'Initial',
            #     path=os.path.join(FLAGS.plots_directory,
            #                       'initial_credit_distribution.png')
            #     if FLAGS.plots_directory else None,
            #     include_median=True,
            #     figure=fig)
            # fig.savefig("a.png")
            #
            # # Initial and final credit distributions next to each other.
            # fig = plt.figure(figsize=(8, 4))
            # plt.subplot(1, 2, 1)
            # lending_plots.plot_credit_distribution(
            #     metrics['initial_credit_distribution'],
            #     'Initial',
            #     path=None,
            #     include_median=True,
            #     figure=fig)
            # plt.subplot(1, 2, 2)
            # fig.savefig("b.png")
            #
            # lending_plots.plot_credit_distribution(
            #     metrics['final_credit_distributions'],
            #     'Final - %s' % title,
            #     path=os.path.join(FLAGS.plots_directory, 'final_credit_distribution.png')
            #     if FLAGS.plots_directory else None,
            #     include_median=True,
            #     figure=fig)
            # fig.savefig("c.png")
            #
            # fig = plt.figure()
            # lending_plots.plot_bars(
            #     metrics['recall'],
            #     title='Recall - %s' % title,
            #     path=os.path.join(FLAGS.plots_directory, 'recall.png')
            #     if FLAGS.plots_directory else None,
            #     figure=fig)
            # fig.savefig("d.png")
            #
            # fig = plt.figure()
            # lending_plots.plot_bars(
            #     metrics['precision'],
            #     title='Precision - %s' % title,
            #     ylabel='Precision',
            #     path=os.path.join(FLAGS.plots_directory, 'precision.png')
            #     if FLAGS.plots_directory else None,
            #     figure=fig)
            # fig.savefig("e.png")
            #
            # fig = plt.figure()
            # lending_plots.plot_cumulative_loans(
            #     {'demo - %s' % title: metrics['cumulative_loans']},
            #     path=os.path.join(FLAGS.plots_directory, 'cumulative_loans.png')
            #     if FLAGS.plots_directory else None,
            #     figure=fig)
            # fig.savefig("f.png")

            # print('Profit %s %f' % (title, result['metric_results']['profit rate']))
            # plt.show()

        if FLAGS.hparams_outfile:
            hparams_df.set_index("sample_id")
            hparams_df.to_csv(FLAGS.hparams_outfile)

    if FLAGS.outfile and not FLAGS.sampling_flag:
        # Save raw data
        with open(FLAGS.outfile, 'w') as f:
            f.write(result_json)

    if FLAGS.metrics_outfile:
        # Save metrics data
        metrics_df.to_csv(FLAGS.metrics_outfile)


if __name__ == '__main__':
    app.run(main)
