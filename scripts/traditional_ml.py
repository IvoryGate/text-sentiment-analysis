"""运行传统机器学习验证集实验。"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sentiment_analysis.evaluation import classification_metrics
from sentiment_analysis.paths import CLEANED_DIR, TABLES_DIR, TFIDF_DIR


def main():
    # 加载清理后的训练集和验证集
    train_data = pd.read_csv(CLEANED_DIR / "train.csv")
    validation_data = pd.read_csv(CLEANED_DIR / "val.csv")

    X_train = train_data["text"]
    y_train = train_data["label"]
    X_validation = validation_data["text"]
    y_validation = validation_data["label"]

    # 使用已拟合的 TF-IDF 转换器生成特征
    feature_sets = {}
    for name in ["unigram", "bigram", "combined"]:
        vectorizer = joblib.load(TFIDF_DIR / f"{name}.joblib")
        feature_sets[name] = {
            "train": vectorizer.transform(X_train),
            "validation": vectorizer.transform(X_validation),
        }

    # 按照 Notebook 中的六组配置训练和评估
    experiments = [
        ("B01", "unigram", "Unigram", "Logistic Regression"),
        ("B02", "bigram", "Bigram", "Logistic Regression"),
        ("B03", "combined", "Unigram + Bigram", "Logistic Regression"),
        ("B04", "unigram", "Unigram", "Linear SVM"),
        ("B05", "bigram", "Bigram", "Linear SVM"),
        ("B06", "combined", "Unigram + Bigram", "Linear SVM"),
    ]

    results = []
    predictions = {}
    for experiment, feature_name, feature_label, model_name in experiments:
        if model_name == "Logistic Regression":
            model = LogisticRegression(C=1.0, max_iter=1000)
        else:
            model = LinearSVC(C=1.0, max_iter=1000, random_state=42)

        model.fit(feature_sets[feature_name]["train"], y_train)
        prediction = model.predict(feature_sets[feature_name]["validation"])
        predictions[experiment] = prediction
        results.append({
            "EXPERIMENT": experiment,
            "FEATURES": feature_label,
            "MODEL": model_name,
            **classification_metrics(y_validation, prediction),
        })

    experiment_results = pd.DataFrame(results)

    # 比较 B01 与 B06 的预测
    b01_correct = predictions["B01"] == y_validation.to_numpy()
    b06_correct = predictions["B06"] == y_validation.to_numpy()
    comparison_summary = pd.DataFrame([
        {"CASE": "Both correct", "COUNT": (b01_correct & b06_correct).sum()},
        {"CASE": "Only B01 correct", "COUNT": (b01_correct & ~b06_correct).sum()},
        {"CASE": "Only B06 correct", "COUNT": (~b01_correct & b06_correct).sum()},
        {"CASE": "Both wrong", "COUNT": (~b01_correct & ~b06_correct).sum()},
    ])

    # 比较组合特征下 Linear SVM 的 C 值
    c_results = []
    for c_value in [0.1, 1.0, 10.0]:
        model = LinearSVC(C=c_value, max_iter=1000, random_state=42)
        model.fit(feature_sets["combined"]["train"], y_train)
        prediction = model.predict(feature_sets["combined"]["validation"])
        c_results.append({
            "C": c_value,
            **classification_metrics(y_validation, prediction),
            "ITERATIONS": model.n_iter_,
        })

    c_comparison = pd.DataFrame(c_results)

    # 保存验证集结果
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    experiment_results.to_csv(TABLES_DIR / "traditional_ml_validation_metrics.csv", index=False)
    comparison_summary.to_csv(TABLES_DIR / "traditional_ml_prediction_comparison.csv", index=False)
    c_comparison.to_csv(TABLES_DIR / "traditional_ml_svm_c_comparison.csv", index=False)

    print("Validation metrics:")
    print(experiment_results.round(4).to_string(index=False))
    print("\nSVM C comparison:")
    print(c_comparison.round(4).to_string(index=False))
    print(f"\nResults saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
