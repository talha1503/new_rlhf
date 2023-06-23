import torch
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F

def preference_loss_function(sum_reward1, sum_reward2, pref):
    multiplicative_factor = (2 * pref) - 1
    score_diff = multiplicative_factor * (sum_reward2-sum_reward1)
    return -torch.mean(torch.log(torch.sigmoid(score_diff)))

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * log(1e-15 + predicted[i])
        mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def preference_loss_function_2(sum_a, sum_b, decisions):
    '''
    sum_a -> batch_size, 1
    sum_b -> batch_size, 1
    '''
    stacked_tensor = torch.cat([sum_a, sum_b], dim=1)
    stacked_tensor = stacked_tensor.to(torch.float32)
    loss = F.cross_entropy(stacked_tensor, decisions)
    return loss
