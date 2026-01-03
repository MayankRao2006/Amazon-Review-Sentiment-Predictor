# 🛍️ Amazon Review Sentiment Predictor

## Overview
**Amazon Review Sentiment Predictor** is a machine learning project designed to classify Amazon product reviews as **positive** or **negative** based on both textual and numerical features. This project demonstrates a production-ready ML pipeline using **scikit-learn**, combining text preprocessing, feature engineering, and model training into a single reusable workflow.

The project handles both unstructured text (review title and text) and structured numerical features (ratings, verified purchase) to improve predictive performance. It showcases practical ML skills including **custom transformers, pipelines, cross-validation, and model persistence**.

---

## Features ✨
- **Custom Text Transformer:** Combines review title and review text into a single feature for TF-IDF vectorization.
- **TF-IDF Vectorization:** Captures unigrams and bigrams from review text for better context understanding.
- **Logistic Regression Classifier:** Efficient and interpretable model for high-dimensional text data.
- **Full Scikit-learn Pipeline:** Integrates preprocessing, feature engineering, and model training for reproducibility and inference.
- **Stratified Train-Test Split:** Maintains class balance during model evaluation.
- **Joblib Model Persistence:** Allows saving and loading of the trained pipeline for inference.
- **Cross-Validation:** Ensures reliable evaluation of model performance.

---

## Dataset 🗂️
- **Source:** (https://www.kaggle.com/datasets/fawadhossaini1415/amazon-fashion-800k-user-reviews-dataset) Amazon Fashion Reviews Dataset 
- **Size:** 800,000+ reviews
- **Columns Used:** `title`, `text`, `rating`, `verified_purchase`
- **Target:** Binary sentiment (positive / negative)
- **Note:** The full dataset is **not included** due to size constraints. Users can download it from Kaggle and run the notebook to reproduce results.

---

## Installation 🛠️
Clone the repository and install the required dependencies:

```bash
1. open main.ipynb file
2. copy paste the code in model.py 

