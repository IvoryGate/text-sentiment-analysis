"""
Download the Large Movie Review Dataset from Hugging Face Datasets. 
https://huggingface.co/datasets/stanfordnlp/imdb
"""
from sentiment_analysis.paths import RAW_DIR
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")

RAW_DIR.mkdir(parents=True, exist_ok=True)

dataset["train"].to_pandas().to_csv(RAW_DIR / "train.csv", index= False, encoding= "utf-8")
dataset["test"].to_pandas().to_csv(RAW_DIR / "test.csv", index= False, encoding= "utf-8")
dataset["unsupervised"].to_pandas().to_csv(RAW_DIR / "unsupervised.csv", index= False, encoding= "utf-8")