import pandas as pd
from tqdm.auto import tqdm

def get_features_target(preference_df, feature_cols, target_cols=['preference_a', 'preference_b']):
    """ Ensure the datatype of each column is set as float.
    """
    features = preference_df[feature_cols].copy()
    for c in feature_cols:
        features[c] = features[c].apply(float)

    features = features.to_numpy()
    decisions = preference_df[target_cols].to_numpy()
    
    return features, decisions

def parse_preference_df(preferences_df, drop_cols=["credit_score_distribution"]):
    """ The ground truth label (labels) is the human feedback (0 for chosen and 1 for rejected).
    """
    reward_modelling_df = pd.DataFrame()
    res_cols = set(["-".join(i.split('-')[1:]) for i in preferences_df.columns if ('option' in i)])
    res_cols = set([i for i in res_cols if (i.split('-')[0] not in drop_cols)])
    for idx, row in preferences_df.iterrows():
        tmp_cols_a = [f"option_a-{i}" for i in res_cols]
        tmp_cols_b = [f"option_b-{i}" for i in res_cols]

        tmp_df_a = pd.DataFrame(row[tmp_cols_a]).T
        tmp_df_b = pd.DataFrame(row[tmp_cols_b]).T
        tmp_df_a.columns = res_cols
        tmp_df_b.columns = res_cols

        tmp_df_a['label'] = 0 if row['preference'][0] == 1 else 1
        tmp_df_b['label'] = 0 if row['preference'][1] == 1 else 1

        reward_modelling_df = reward_modelling_df.append(tmp_df_a)
        reward_modelling_df = reward_modelling_df.append(tmp_df_b)
        
    reward_modelling_df = reward_modelling_df.reset_index(drop=True)
    
    return reward_modelling_df
