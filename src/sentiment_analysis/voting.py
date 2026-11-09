"""固定三模型投票与测试集指标。"""
import numpy as np
import pandas as pd

from sentiment_analysis.evaluation import classification_metrics
from sentiment_analysis.inference import load_fixed_test_predictions
from sentiment_analysis.paths import CONFIGS_DIR


VOTING_PROTOCOL_PATH = CONFIGS_DIR / "three_model_voting_test_protocol.json"


def evaluate_fixed_voting(protocol):
    """返回硬投票和软投票的指标表与逐条预测。"""
    labels, model_predictions, model_probabilities = load_fixed_test_predictions(protocol)
    positive_votes = (model_predictions.to_numpy() == protocol["positive_label"]).sum(axis=1)
    hard_vote = (positive_votes >= protocol["hard_min_positive_votes"]).astype(int)
    predictions = {"Hard vote": hard_vote}
    hard_correct = hard_vote == labels

    probabilities = model_probabilities.to_numpy()
    soft_probabilities = {}
    threshold = protocol["soft_positive_threshold"]
    for name, weights in protocol["soft_weights"].items():
        weights = np.array(weights)
        if weights.shape != (probabilities.shape[1],) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError(f"Invalid weights: {name}")
        soft_probabilities[name] = probabilities @ weights
        predictions[name] = (soft_probabilities[name] > threshold).astype(int)

    hard_metrics = classification_metrics(labels, hard_vote)
    rows = []
    for name, prediction in predictions.items():
        metrics = hard_metrics if name == "Hard vote" else classification_metrics(labels, prediction)
        correct = prediction == labels
        rows.append({
            "MODEL": name, **metrics,
            "F1 CHANGE VS HARD (PP)": 100 * (metrics["F1"] - hard_metrics["F1"]),
            "CORRECTED VS HARD": int((~hard_correct & correct).sum()),
            "NEW ERRORS VS HARD": int((hard_correct & ~correct).sum()),
        })

    prediction_table = pd.DataFrame({
        "TEST ROW": np.arange(len(labels)),
        "LABEL": labels,
        **{f"{name.upper()} POSITIVE PROBABILITY": values for name, values in model_probabilities.items()},
        **dict(model_predictions.items()),
        **{f"{name.upper()} POSITIVE PROBABILITY": scores for name, scores in soft_probabilities.items()},
        **predictions,
    })
    return pd.DataFrame(rows), prediction_table


def save_voting_results(output_dir, comparison, predictions):
    """使用 Notebook 023 的 CRLF 格式保存结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "three_model_voting_test_comparison.csv", index=False, lineterminator="\r\n")
    predictions.to_csv(output_dir / "three_model_voting_test_predictions.csv", index=False, lineterminator="\r\n")
