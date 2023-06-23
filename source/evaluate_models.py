import torch
from sklearn import metrics
import pandas as pd
import numpy as np

def evaluate_model(model, test_features, test_targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_features = torch.tensor(test_features, dtype=torch.float32, device=device)
    input_dim = test_features.shape[1]//2
    
    with torch.inference_mode():
        reward_A = model(test_features[:, :input_dim])
        reward_B = model(test_features[:, input_dim:])
    
    y_pred = torch.gt(reward_B.detach().cpu().float(), reward_A.detach().cpu().float())
    y_pred = y_pred.int().tolist()
    
    cm = metrics.confusion_matrix(test_targets, y_pred)
#     metrics_report = metrics.classification_report(test_targets, y_pred, output_dict=True)
#     metrics_report_df = pd.DataFrame(metrics_report).transpose()
    clf_rep = metrics.precision_recall_fscore_support(test_targets, y_pred)
    #Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #The diagonal entries are the accuracies of each class
    class_acc = np.array(cm.diagonal())
    cm_df = pd.DataFrame(cm)
    out_dict = {"precision" :clf_rep[0].round(2),
                 "recall" : clf_rep[1].round(2),
                 "f1-score" : clf_rep[2].round(2),
                 "support" : clf_rep[3],
                 "accuracy": class_acc.round(2)}
    
    out_df = pd.DataFrame(out_dict)
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    
    return out_df, cm_df


def evaluate_model_new(model, test_loader, model_input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_pred = []
    y_true = []
#     with torch.no_grad():
    with torch.inference_mode():
        for test_features, y in test_loader:
            reward_A = model(test_features[:, :, :model_input_dim])
            reward_B = model(test_features[:, :, model_input_dim:])
            tmp_y_pred = list(map(float, torch.gt(reward_B.detach().cpu().float(), reward_A.detach().cpu().float()).tolist()))
            y_pred += tmp_y_pred
            y_true += y.tolist()
    
    clf_rep = metrics.precision_recall_fscore_support(y_true, y_pred)
    cm_df = pd.DataFrame(metrics.confusion_matrix(y_true, y_pred))
    out_dict = {
                 "precision" :clf_rep[0].round(2),
                 "recall" : clf_rep[1].round(2),
                 "f1-score" : clf_rep[2].round(2),
                 "support" : clf_rep[3],
                 "accuracy": class_acc.round(2)
                }
    
    out_df = pd.DataFrame(out_dict)
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    return out_df, cm_df
