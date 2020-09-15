import numpy as np
import pandas as pd

def accuracy_score(y_target, y_predict):
    if type(y_target)==pd.core.frame.DataFrame:
        y_target = y_target.values
    
    return np.mean(y_target==y_predict)

def precision_score(y_target, y_predict):
    if type(y_target)==pd.core.frame.DataFrame:
        y_target = y_target.values

    TP = ((y_target * y_predict) == 1).sum()
    P_predict = (y_predict == 1).sum()

    return TP / P_predict

    
def recall_score(y_target, y_predict):
    if type(y_target)==pd.core.frame.DataFrame:
        y_target = y_target.values

    TP = ((y_target * y_predict) == 1).sum()
    P_target = (y_target == 1).sum()
    
    return TP / P_target

def f1_score(y_target, y_predict):
    recall = recall_score(y_target, y_predict)
    precision = precision_score(y_target, y_predict)
    f1 = 2*recall*precision / (recall + precision)
    return f1