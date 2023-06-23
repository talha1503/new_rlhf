import os
import numpy as np
import pandas as pd
import bz2
import itertools

# reference: https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0
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


class CompressionBasedDissimilarity:

    def __init__(self, n_letters=7):
        self.bins = None
        self.n_letters = n_letters

    def set_bins(self, bins):
        self.bins = bins

    def sax_bins(self, all_values):
        bins = np.percentile(
            all_values[all_values > 0], np.linspace(0, 100, self.n_letters + 1)
        )
        bins[0] = 0
        bins[-1] = 1e1000
        return bins

    @staticmethod
    def sax_transform(all_values, bins):
        indices = np.digitize(all_values, bins) - 1
        alphabet = np.array([*("abcdefghijklmnopqrstuvwxyz"[:len(bins) - 1])])
        text = "".join(alphabet[indices])
        return str.encode(text)

    def calculate(self, m, n):
        if self.bins is None:
            m_bins = self.sax_bins(m)
            n_bins = self.sax_bins(n)
        else:
            m_bins = n_bins = self.bins
        m = self.sax_transform(m, m_bins)
        n = self.sax_transform(n, n_bins)
        len_m = len(bz2.compress(m))
        len_n = len(bz2.compress(n))
        len_combined = len(bz2.compress(m + n))
        return len_combined / (len_m + len_n)


def calculate_smilarity(series1, series2):
    euclidean_dist = calc_euclidean(series1, series2)
    mape_dist = calc_mape(series1, series2)
    correlation_dist = calc_correlation(series1, series2)
    # compression_similarity_class = CompressionBasedDissimilarity()
    # compression_similarity_dist = compression_similarity_class.calculate(series1, series2)
    return {
        'euclidean_distance':euclidean_dist,
        'mape_dist':mape_dist,
        'correlation_dist':correlation_dist,
        # 'compression_similarity_dist':compression_similarity_dist
    }

if __name__ == '__main__':
    fair_dir = 'D:\Work\EleutherAI\\fairness_gym\ml-fairness-gym\data\\fair_trajectories'
    greedy_dir = 'D:\Work\EleutherAI\\fairness_gym\ml-fairness-gym\data\\greedy_trajectories'

    fair_files = os.listdir(fair_dir)
    greedy_files = os.listdir(greedy_dir)

    pairings = list(itertools.product(fair_files, greedy_files))
    for fair_file, greedy_file in pairings:
        fair_df = pd.read_csv(os.path.join(fair_dir, fair_file))
        greedy_df = pd.read_csv(os.path.join(greedy_dir, greedy_file))

        print("Average Credit Score: ")
        similarity_dict_1 = calculate_smilarity(fair_df['average_credit_score-group_1'], greedy_df['average_credit_score-group_1'])
        similarity_dict_2 = calculate_smilarity(fair_df['average_credit_score-group_2'], greedy_df['average_credit_score-group_2'])
        print(similarity_dict_1)
        print(similarity_dict_2)

        print("Acceptance Rate: ")
        similarity_dict_1 = calculate_smilarity(fair_df['acceptance_rate-group_1'],
                                                greedy_df['acceptance_rate-group_1'])
        similarity_dict_2 = calculate_smilarity(fair_df['acceptance_rate-group_2'],
                                                greedy_df['acceptance_rate-group_2'])
        print(similarity_dict_1)
        print(similarity_dict_2)
        print("-"*50)