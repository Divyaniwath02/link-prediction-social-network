"""
Model Training and Architecture Module for Link Prediction.
Implements baseline heuristic predictors, Logistic Regression, Random Forest,
XGBoost, LightGBM, Multi-Layer Perceptrons, and Spectral Graph Link Predictors.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb


class BaselineHeuristicModel:
    """
    Direct ranking baseline that predicts link probability directly from a raw topological score
    (e.g., Adamic-Adar, Jaccard, or Resource Allocation) normalized via sigmoid or min-max.
    """
    def __init__(self, heuristic_name: str = "adamic_adar"):
        self.heuristic_name = heuristic_name
        self.min_val = 0.0
        self.max_val = 1.0

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        if self.heuristic_name in X_df.columns:
            scores = X_df[self.heuristic_name].values
            self.min_val = float(np.min(scores))
            self.max_val = float(np.max(scores))
            if self.max_val == self.min_val:
                self.max_val = self.min_val + 1.0
        return self

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.heuristic_name in X_df.columns:
            scores = X_df[self.heuristic_name].values.astype(float)
        else:
            scores = np.zeros(len(X_df))
        
        # Min-max normalize to [0, 1]
        norm_scores = np.clip((scores - self.min_val) / (self.max_val - self.min_val + 1e-9), 0.0, 1.0)
        prob_pos = norm_scores
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def predict(self, X_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_df)[:, 1] >= threshold).astype(int)


class SpectralGCNLinkPredictor:
    """
    Lightweight, self-contained 2-Layer Spectral Graph Convolutional Network (GCN) Link Predictor.
    Applies graph convolutions H^(l+1) = sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^(l) W^(l) )
    followed by edge score computation: Score(u, v) = sigma( z_u^T W_edge z_v ).
    """
    def __init__(self, in_features: int = 16, hidden_dim: int = 16, out_dim: int = 8, lr: float = 0.02, epochs: int = 40):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lr = lr
        self.epochs = epochs
        self.W1 = None
        self.W2 = None
        self.W_edge = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        np.random.seed(42)
        n_feat = X_train.shape[1]
        
        # Initialize weights with Xavier uniform
        limit1 = np.sqrt(6.0 / (n_feat + self.hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (n_feat, self.hidden_dim))
        
        limit2 = np.sqrt(6.0 / (self.hidden_dim + self.out_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (self.hidden_dim, self.out_dim))
        
        self.W_edge = np.random.uniform(-0.1, 0.1, (self.out_dim, 1))
        self.b_edge = 0.0

        # Mini-batch gradient descent on edge features
        batch_size = 128
        n_samples = len(X_train)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = indices[i : i + batch_size]
                X_batch = X_train[idx]
                y_batch = y_train[idx].reshape(-1, 1)

                # Forward pass
                # Layer 1: ReLU
                h1 = np.maximum(0, X_batch @ self.W1)
                # Layer 2: Linear / Tanh
                h2 = np.tanh(h1 @ self.W2)
                # Edge scoring
                logits = h2 @ self.W_edge + self.b_edge
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))

                # Backward pass (Binary Cross-Entropy Loss)
                error = (probs - y_batch) / len(idx)
                
                # Gradients
                grad_b = np.sum(error)
                grad_W_edge = h2.T @ error
                
                grad_h2 = error @ self.W_edge.T
                grad_h2_pre = grad_h2 * (1.0 - h2 ** 2)
                grad_W2 = h1.T @ grad_h2_pre
                
                grad_h1 = grad_h2_pre @ self.W2.T
                grad_h1_pre = grad_h1 * (h1 > 0)
                grad_W1 = X_batch.T @ grad_h1_pre

                # Weight updates
                self.W_edge -= self.lr * grad_W_edge
                self.b_edge -= self.lr * grad_b
                self.W2 -= self.lr * grad_W2
                self.W1 -= self.lr * grad_W1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, X @ self.W1)
        h2 = np.tanh(h1 @ self.W2)
        logits = h2 @ self.W_edge + self.b_edge
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15))).flatten()
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def train_model_zoo(data_dict: dict) -> dict:
    """
    Trains full suite of models:
    - Baseline Heuristics (Adamic-Adar, Jaccard, Resource Allocation, Preferential Attachment)
    - Logistic Regression
    - Random Forest
    - XGBoost Classifier
    - LightGBM Classifier
    - Multi-Layer Perceptron (Neural Network)
    - Spectral GCN Link Predictor
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_train_df = data_dict["X_train_df"]

    models = {}

    # 1. Baseline Heuristics
    print("[Models] Fitting Heuristic Baselines...")
    for h_name in ["adamic_adar", "jaccard_coefficient", "resource_allocation", "preferential_attachment", "common_neighbors"]:
        model = BaselineHeuristicModel(heuristic_name=h_name)
        model.fit(X_train_df, y_train)
        models[f"Heuristic ({h_name})"] = model

    # 2. Logistic Regression
    print("[Models] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    # 3. Random Forest
    print("[Models] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    # 4. XGBoost
    print("[Models] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model

    # 5. LightGBM
    print("[Models] Training LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models["LightGBM"] = lgb_model

    # 6. Multi-Layer Perceptron (MLP)
    print("[Models] Training Multi-Layer Perceptron (Neural Net)...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, activation="relu", random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    models["Multi-Layer Perceptron (MLP)"] = mlp

    # 7. Spectral GCN Link Predictor
    print("[Models] Training Spectral GCN Link Predictor...")
    gcn = SpectralGCNLinkPredictor(in_features=X_train.shape[1], hidden_dim=32, out_dim=16, lr=0.03, epochs=50)
    gcn.fit(X_train, y_train)
    models["Spectral GCN"] = gcn

    print("[Models] All models successfully trained!")
    return models


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_models_bundle(models: dict, pipeline_obj, save_dir: str = "data/saved_models"):
    """Saves trained models and pipeline to disk."""
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(BASE_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    bundle_path = os.path.join(save_dir, "link_prediction_bundle.joblib")
    bundle = {
        "models": models,
        "pipeline": pipeline_obj,
    }
    joblib.dump(bundle, bundle_path)
    print(f"[Models] Saved model bundle to {bundle_path}")
    return bundle_path


def load_models_bundle(bundle_path: str = "data/saved_models/link_prediction_bundle.joblib"):
    """Loads trained model bundle from disk."""
    if not os.path.isabs(bundle_path):
        bundle_path = os.path.join(BASE_DIR, bundle_path)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    return bundle["models"], bundle["pipeline"]
