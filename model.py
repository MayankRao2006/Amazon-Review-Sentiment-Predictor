import pandas as pd
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "full_pipeline.pkl"

# Building a reusable class which will combine text columns
class Text_combiner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            X[self.columns].fillna("").agg(" ".join,axis=1)
        )


if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("amazon-fashion-800k+-user-reviews-dataset.csv")
    df.dropna(inplace=True)

    # Stratified Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["rating"]):
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]
    
    cols_to_drop = ["asin", "parent_asin", "images", "user_id", "timestamp"]
    train_set = train_set.drop(cols_to_drop, axis=1)
    test_set = test_set.drop(cols_to_drop, axis=1)

    test_set.to_csv("Testing_data.csv", index=False)
    
    features = train_set.drop("target", axis=1)
    labels = train_set["target"].copy()

    num_attrs = features.drop(["title", "text"], axis=1).columns
    cat_attrs = ["title", "text"]

    text_pipeline = Pipeline([
        ("text_combiner", Text_combiner(["title", "text"])),
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1,2), stop_words="english", min_df=5))
    ])

    preprocessor = ColumnTransformer([
        ("text_handling", text_pipeline, cat_attrs),
        ("num_pipeline", "passthrough", num_attrs)
    ])

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    full_pipeline.fit(features, labels)
    joblib.dump(full_pipeline, MODEL_FILE)

    print("Full Pipeline saved successfully.")

else:
    # Inference phase
    full_pipeline = joblib.load(MODEL_FILE)

    df = pd.read_csv("Testing_data.csv")

    if "target" in df.columns:
        df = df.drop("target", axis=1)

    predictions = full_pipeline.predict(df)
    df["predictions"] = predictions

    df.to_csv("Predictions.csv", index=False)
    print("Inference saved to Predictions.csv")
