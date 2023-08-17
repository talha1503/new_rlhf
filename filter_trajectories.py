import os
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import itertools
import json
from tqdm import tqdm

def calc_euclidean(series1, series2):
    return np.sqrt(np.sum((series1 - series2) ** 2))


def calc_mape(series1, series2):
    return np.mean(np.abs((series1 - series2) / series2))


def calc_correlation(series1, series2):
    series1_diff = series1 - np.mean(series1)
    series2_diff = series2 - np.mean(series2)
    numerator = np.sum(series1_diff * series2_diff)
    denominator = np.sqrt(np.sum(series1_diff ** 2)) * np.sqrt(np.sum(series2_diff ** 2))
    return numerator / denominator

def dtw_distance_metric(series1, series2, window_size):
    n = len(series1)
    m = len(series2)

    # Create a matrix to store the accumulated distances
    dtw_matrix = np.zeros((n + 1, m + 1))

    # Set initial values
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf

    # Calculate the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            window = abs(i - j) <= window_size
            if window:
                cost = abs(series1[i - 1] - series2[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    # Return the DTW distance
    return dtw_matrix[n, m]

def min_max_scale(series):
    min_val = np.min(series)
    max_val = np.max(series)
    scaled_series = (series - min_val) / (max_val - min_val)
    return scaled_series

def mean_scale(series1, series2):
    mean_val = np.mean([np.mean(series1), np.mean(series2)])
    scaled_series1 = series1 / np.mean(series1) * mean_val
    scaled_series2 = series2 / np.mean(series2) * mean_val
    return scaled_series1, scaled_series2

def calculate_smilarity(series1, series2):
    euclidean_dist = calc_euclidean(series1, series2)
    mape_dist = calc_mape(series1, series2)
    correlation_dist = calc_correlation(series1, series2)
    dtw_distance = dtw_distance_metric(series1, series2, 50)

    series1, series2 = mean_scale(series1, series2)

    dtw_distance_scaled = dtw_distance_metric(series1, series2, 50)
    euclidean_dist_scaled = calc_euclidean(series1, series2)
    mape_dist_scaled = calc_mape(series1, series2)
    correlation_dist_scaled = calc_correlation(series1, series2)
    # compression_similarity_class = CompressionBasedDissimilarity()
    # compression_similarity_dist = compression_similarity_class.calculate(series1, series2)
    return {
        # 'euclidean_distance':'%.3f'% (euclidean_dist),
        # 'mape_dist':'%.3f'% (mape_dist),
        # 'correlation_dist':'%.3f'% (correlation_dist),
        # 'dtw_distance':'%.3f'% (dtw_distance),
        # 'euclidean_distance_scaled':'%.3f'% (euclidean_dist_scaled),
        # 'mape_dist_scaled':'%.3f'% (mape_dist_scaled),
        # 'correlation_dist_scaled':'%.3f'% (correlation_dist_scaled),
        'dtw_distance_scaled':'%.3f'% (dtw_distance_scaled),
        # 'compression_similarity_dist':compression_similarity_dist
    }


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (10,10)

    base_dir = './data'
    image_dir = 'new_comparisons_15'

    greedy_directories = ['greedy_cleanrl_7',]
    fair_trajectories = ['ppo_fair_trajectories_new_2']

    directory_pairings = list(itertools.product(fair_trajectories, greedy_directories))

    for dir_index, (fair_dir, greedy_dir) in tqdm(enumerate(directory_pairings)): 
        # fair_dir = 'fair_trajectories_2'
        # greedy_dir = 'greedy_trajectories'

        fair_files = os.listdir(os.path.join(base_dir, fair_dir))
        greedy_files = os.listdir(os.path.join(base_dir, greedy_dir))
        
        fair_files = [file for file in fair_files if file!='.ipynb_checkpoints']
        greedy_files = [file for file in greedy_files if file!='.ipynb_checkpoints']

        pairings = list(itertools.product(fair_files, greedy_files))
        if not os.path.isdir(os.path.join(base_dir, image_dir, fair_dir + '_' + greedy_dir)):
            os.makedirs(os.path.join(base_dir, image_dir, str(fair_dir + '_' + greedy_dir)))
        else:
            continue

        # results_file = open(os.path.join(base_dir, image_dir, fair_dir + '_' + greedy_dir,fair_dir + '_'+ greedy_dir +'.txt'),"a")
        for index, (fair_file, greedy_file) in enumerate(pairings):
            fair_df = pd.read_csv(os.path.join(base_dir, fair_dir, fair_file))
            greedy_df = pd.read_csv(os.path.join(base_dir, greedy_dir, greedy_file))

            plt.subplot(4,2,1)
            plt.plot(fair_df["Timestep"],fair_df["acceptance_rate-group_1"],label = "group1")
            plt.plot(fair_df["Timestep"],fair_df["acceptance_rate-group_2"], label = "group2")
            plt.title("Fair trajectory Acceptance Rate")
            plt.legend()

            plt.subplot(4,2,2)
            plt.plot(greedy_df["Timestep"],greedy_df["acceptance_rate-group_1"],label = "group1")
            plt.plot(greedy_df["Timestep"],greedy_df["acceptance_rate-group_2"], label = "group2")
            plt.title("Greedy trajectory Acceptance Rate")
            plt.legend()

            plt.subplot(4,2,3)
            plt.plot(fair_df["Timestep"],fair_df["average_credit_score-group_1"],label = "group1")
            plt.plot(fair_df["Timestep"],fair_df["average_credit_score-group_2"], label = "group2")
            plt.title("Fair trajectory Avg credit score")
            plt.legend()

            plt.subplot(4,2,4)
            plt.plot(greedy_df["Timestep"],greedy_df["average_credit_score-group_1"],label = "group1")
            plt.plot(greedy_df["Timestep"],greedy_df["average_credit_score-group_2"],label = "group2")
            plt.title("Greedy trajectory Avg credit score")

            plt.subplot(4,2,5)
            plt.plot(fair_df["Timestep"],fair_df["default_rate-group_1"],label = "group1")
            plt.plot(fair_df["Timestep"],fair_df["default_rate-group_2"], label = "group2")
            plt.title("Fair trajectory Default rate")
            plt.legend()

            plt.subplot(4,2,6)
            plt.plot(greedy_df["Timestep"],greedy_df["default_rate-group_1"],label = "group1")
            plt.plot(greedy_df["Timestep"],greedy_df["default_rate-group_2"], label = "group2")
            plt.title("Greedy trajectory Default rate")

            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(base_dir, image_dir, fair_dir + '_' + greedy_dir, fair_file+'_'+greedy_file+'.png'))
            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

#             results_file.write(fair_file+'_'+greedy_file)
#             results_file.write("\n")
#             similarity_dict_1 = calculate_smilarity(fair_df['average_credit_score-group_1'], greedy_df['average_credit_score-group_1'])
#             similarity_dict_2 = calculate_smilarity(fair_df['average_credit_score-group_2'], greedy_df['average_credit_score-group_2'])
#             results_file.write("Average Credit Score: ")
#             results_file.write("\n")
#             results_file.write("Group1: ")
#             results_file.write("\n")
#             results_file.write(json.dumps(similarity_dict_1))
#             results_file.write("\n")
#             results_file.write("Group2: ")
#             results_file.write("\n")
#             results_file.write(json.dumps(similarity_dict_2))
#             results_file.write("\n")
#             # print("Group1: ", similarity_dict_1)
#             # print("Group2: ", similarity_dict_2)

#             similarity_dict_1 = calculate_smilarity(fair_df['acceptance_rate-group_1'],
#                                                     greedy_df['acceptance_rate-group_1'])
#             similarity_dict_2 = calculate_smilarity(fair_df['acceptance_rate-group_2'],
#                                                     greedy_df['acceptance_rate-group_2'])
#             results_file.write("Acceptance Rate: ")
#             results_file.write("\n")
#             results_file.write("Group1: ")
#             results_file.write("\n")
#             results_file.write(json.dumps(similarity_dict_1))
#             results_file.write("\n")
#             results_file.write("Group2: ")
#             results_file.write(json.dumps(similarity_dict_2))
#             results_file.write("\n")
#             results_file.write("-"*50)
#             results_file.write("\n")
            print("-"*50)
        # results_file.close()