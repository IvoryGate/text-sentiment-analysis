"""情感分类模型的评价指标。"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, y_prediction):
    """计算二分类指标，默认以标签 1 为正类。"""
    return {
        "ACCURACY": accuracy_score(y_true, y_prediction),
        "PRECISION": precision_score(y_true, y_prediction),
        "RECALL": recall_score(y_true, y_prediction),
        "F1": f1_score(y_true, y_prediction),
    }
