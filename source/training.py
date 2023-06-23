# import logging
# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from source.datasets import TabularDataset
#
#
# def train_classification_model(model,
#                                input_dim,
#                                training_features,
#                                training_targets,
#                                loss_function,
#                                learning_rate,
#                                num_epochs,
#                                batch_size,
#                                save_dir,
#                                ):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info("Training with: {}".format(device))
#
#     model.train()
#     model.to(device)
#
#     training_dataset = TabularDataset(
#         features=training_features, targets=training_targets, device=device
#     )
#     training_loader = torch.utils.data.DataLoader(
#         training_dataset, batch_size=batch_size, shuffle=True
#     )
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     losses = []
#     for epoch in range(num_epochs):
#         with tqdm(training_loader) as batches:
#             batches.set_description("Training - Epoch {}".format(epoch + 1))
#             for features, targets in batches:
#                 optimizer.zero_grad()
#                 reward_a = model(features[:,:input_dim])
#                 reward_b = model(features[:, input_dim:])
#                 predictions = model(features[:, :input_dim])
#                 loss = loss_function(reward_a, reward_b, targets.reshape(-1, 1))
#                 loss.backward()
#                 optimizer.step()
#                 losses.append(loss.item())
#                 batches.set_postfix({"loss": loss.item()})
#     torch.save(model, os.path.join(save_dir, model.name + ".pt"))
#     return losses
#
#
# def get_norm_dict(df_train, target):
#     mean_dict = {}
#     stdev_dict = {}
#     for c in df_train.columns:
#         if c != target:
#             mean_dict[c] = df_train[c].mean()
#             stdev_dict[c] = df_train[c].std()
#     return mean_dict, stdev_dict
#
#
# def normalize_df(df, decision_col, mean_dict, stdev_dict):
#     for c in df.columns:
#         if c != decision_col:
#             if c in stdev_dict and stdev_dict[c] != 0:
#                 df[c] = (df[c] - mean_dict[c]) / stdev_dict[c]
#             else:
#                 df[c] = 0
#     return df
#
#
# def get_dataloader(df, features, target, sequence_length, batch_size, torch_seed=101, shuffle=False):
#     # Create dataloader objects
#     torch.manual_seed(torch_seed)
#     dataset = SequenceDataset(df, target=target, features=features, sequence_length=sequence_length)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return data_loader
#
#
# def train_reward_model(model,
#                        input_dim,
#                        train_loader,
#                        loss_function,
#                        learning_rate,
#                        num_epochs,
#                        batch_size,
#                        save_dir,
#                        ):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info("Training with: {}".format(device))
#     model.train()
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     losses = [-np.inf]
#     for epoch in range(num_epochs):
#         tmp_losses = []
#         with tqdm(train_loader) as batches:
#             batches.set_description(f"Training - Epoch {epoch + 1} - Loss {round(losses[-1], 4)}")
#             for features, decisions in batches:
#                 optimizer.zero_grad()
#                 reward_A = model(features[:, :, :input_dim])
#                 reward_B = model(features[:, :, input_dim:])
#                 loss = loss_function(reward_A, reward_B, decisions)
#                 loss.backward()
#                 optimizer.step()
#                 tmp_losses.append(loss.item())
#
#             losses.append(np.mean(tmp_losses))
#             batches.set_postfix({"loss": np.mean(losses)})
#
#     torch.save(model, os.path.join(save_dir, model.name + ".pt"))
#     return losses

import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from source.datasets import TabularDataset

def train_classification_model(model,
                               training_features,
                               training_targets,
                               loss_function,
                               learning_rate,
                               num_epochs,
                               batch_size,
                               save_dir,
                               ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training with: {}".format(device))

    model.train()
    model.to(device)

    training_dataset = TabularDataset(
        features=training_features, targets=training_targets, device=device
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        with tqdm(training_loader) as batches:
            batches.set_description("Training - Epoch {}".format(epoch + 1))
            for features, targets in batches:
                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_function(predictions, targets.reshape(-1, 1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                batches.set_postfix({"loss": loss.item()})
    torch.save(model, os.path.join(save_dir, model.name + ".pt"))
    return losses


def get_norm_dict(df_train, target):
    mean_dict = {}
    stdev_dict = {}
    for c in df_train.columns:
        if c != target:
            mean_dict[c] = df_train[c].mean()
            stdev_dict[c] = df_train[c].std()
    return mean_dict, stdev_dict


def normalize_df(df, decision_col, mean_dict, stdev_dict):
    for c in df.columns:
        if c != decision_col:
            if c in stdev_dict and stdev_dict[c] != 0:
                df[c] = (df[c] - mean_dict[c]) / stdev_dict[c]
            else:
                df[c] = 0
    return df


# def get_dataloader(df, features, target, sequence_length, batch_size, torch_seed=101, shuffle=False):
#     # Create dataloader objects
#     torch.manual_seed(torch_seed)
#     dataset = SequenceDataset(df, target=target, features=features, sequence_length=sequence_length)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return data_loader


def train_reward_model(model,
                       input_dim,
                       train_loader,
                       loss_function,
                       learning_rate,
                       num_epochs,
                       batch_size,
                       save_dir,
                       ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training with: {}".format(device))
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = [-np.inf]
    for epoch in range(num_epochs):
        print("ON EPOCH {}: ".format(epoch))
        tmp_losses = []
        reward_A_list = []
        reward_B_list = []

        for features, decisions in train_loader:
            optimizer.zero_grad()
            features = features.to(device)
            feature_last_dim = features.size(2)
            features_a = features[:, :input_dim, :].view(-1, input_dim).to(device)
            features_b = features[:, input_dim:, :].view(-1, input_dim).to(device)
            reward_A = model(features_a)
            reward_B = model(features_b)
            reward_A = torch.sum(reward_A.view(batch_size, -1), 1)
            reward_B = torch.sum(reward_B.view(batch_size, -1), 1)
            decisions = F.one_hot(decisions.to(torch.int64), num_classes=2)
            loss = loss_function(reward_A, reward_B, decisions)
            loss.backward()
            optimizer.step()
            tmp_losses.append(loss.item())

        losses.append(np.mean(tmp_losses))
            # batches.set_postfix({"loss": np.mean(losses)})

    torch.save(model, os.path.join(save_dir, model.name + ".pt"))
    return losses
