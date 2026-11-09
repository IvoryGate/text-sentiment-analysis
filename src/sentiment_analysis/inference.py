"""加载固定测试集的模型预测与正类概率。"""
import hashlib

import joblib
import numpy as np
import pandas as pd

from sentiment_analysis.paths import ROOT


def load_fixed_test_predictions(protocol):
    """核对固定输入，复用已保存的预测并计算 B06 的校准概率。"""
    files = {}
    for name, record in protocol["files"].items():
        path = ROOT / record["path"]
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Fixed input changed: {name} ({path})")
        files[name] = path

    test_data = pd.read_csv(files["test"])
    labels = test_data["label"].to_numpy()
    test_rows = np.arange(len(test_data))
    positive_label = protocol["positive_label"]
    if test_data.empty or test_data["text"].isna().any() or not np.isin(labels, [0, positive_label]).all():
        raise ValueError("Invalid test data.")

    traditional = pd.read_csv(files["fixed_predictions"], usecols=[
        "TEST ROW", "LABEL", "B06", "TextCNN", "TEXTCNN POSITIVE PROBABILITY",
    ])
    distilbert = pd.read_csv(files["distilbert_predictions"], usecols=[
        "TEST ROW", "LABEL", "DistilBERT", "DISTILBERT POSITIVE PROBABILITY",
    ])
    for predictions in (traditional, distilbert):
        if (not np.array_equal(predictions["TEST ROW"], test_rows)
                or not np.array_equal(predictions["LABEL"], labels)):
            raise ValueError("Saved prediction rows or labels do not match the test split.")

    calibrated_b06 = joblib.load(files["calibrated_b06"])
    positive_index = list(calibrated_b06.classes_).index(positive_label)
    b06_probability = calibrated_b06.predict_proba(test_data["text"])[:, positive_index]

    model_order = protocol["model_order"]
    model_predictions = pd.concat([traditional[["B06", "TextCNN"]], distilbert[["DistilBERT"]]], axis=1)[model_order]
    model_probabilities = pd.DataFrame({
        "B06": b06_probability,
        "TextCNN": traditional["TEXTCNN POSITIVE PROBABILITY"],
        "DistilBERT": distilbert["DISTILBERT POSITIVE PROBABILITY"],
    })[model_order]
    if not np.isin(model_predictions.to_numpy(), [0, positive_label]).all():
        raise ValueError("Invalid model predictions.")
    probabilities = model_probabilities.to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Invalid model probabilities.")
    return labels, model_predictions, model_probabilities
