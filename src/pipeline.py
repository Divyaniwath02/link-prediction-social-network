"""
Feature Pipeline Module for Link Prediction.
Extracts and combines topological heuristics and node embeddings into ML-ready datasets,
ensuring strict separation of G_train to prevent data leakage.
"""

import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.heuristics import precompute_graph_structures, compute_edge_features_fast
from src.node_embeddings import Node2Vec, compute_edge_embedding_features


class LinkPredictionPipeline:
    """
    End-to-end dataset builder that takes graph edge splits and transforms them into
    rich feature matrices (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    def __init__(self, use_embeddings: bool = True, embedding_dim: int = 16, seed: int = 42):
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.scaler = StandardScaler()
        self.feature_names = []
        self.node2vec_model = None
        self.precomputed_structures = None

    def fit_transform_splits(self, splits: dict) -> dict:
        """
        Extracts features on train, val, and test edge sets.
        ALL graph statistics and embeddings are derived exclusively from splits['train_graph'].
        """
        G_train = splits["train_graph"]
        print(f"[Pipeline] Computing graph features on training graph ({G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges)...")
        
        # 1. Precompute graph heuristics structures on G_train
        self.precomputed_structures = precompute_graph_structures(G_train)
        
        # 2. Extract topological features for all splits
        df_train_topo = compute_edge_features_fast(G_train, splits["train_data"], self.precomputed_structures)
        df_val_topo = compute_edge_features_fast(G_train, splits["val_data"], self.precomputed_structures)
        df_test_topo = compute_edge_features_fast(G_train, splits["test_data"], self.precomputed_structures)
        
        # 3. Fit Node2Vec on G_train if enabled
        if self.use_embeddings:
            self.node2vec_model = Node2Vec(dimensions=self.embedding_dim, walk_length=15, num_walks=8, seed=self.seed)
            self.node2vec_model.fit(G_train)
            
            df_train_emb = compute_edge_embedding_features(self.node2vec_model.embeddings, splits["train_data"])
            df_val_emb = compute_edge_embedding_features(self.node2vec_model.embeddings, splits["val_data"])
            df_test_emb = compute_edge_embedding_features(self.node2vec_model.embeddings, splits["test_data"])
            
            # Merge topological and embedding features
            df_train_full = pd.concat([df_train_topo, df_train_emb], axis=1)
            df_val_full = pd.concat([df_val_topo, df_val_emb], axis=1)
            df_test_full = pd.concat([df_test_topo, df_test_emb], axis=1)
        else:
            df_train_full = df_train_topo
            df_val_full = df_val_topo
            df_test_full = df_test_topo

        # Labels
        y_train = np.array([p[2] for p in splits["train_data"]], dtype=int)
        y_val = np.array([p[2] for p in splits["val_data"]], dtype=int)
        y_test = np.array([p[2] for p in splits["test_data"]], dtype=int)

        # Drop identifiers u and v from feature matrix
        feature_cols = [c for c in df_train_full.columns if c not in ["u", "v"]]
        self.feature_names = feature_cols

        X_train_df = df_train_full[feature_cols].copy()
        X_val_df = df_val_full[feature_cols].copy()
        X_test_df = df_test_full[feature_cols].copy()

        # Handle any NaN/Inf values
        X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_df.values)
        X_val_scaled = self.scaler.transform(X_val_df.values)
        X_test_scaled = self.scaler.transform(X_test_df.values)

        print(f"[Pipeline] Feature extraction complete: {len(feature_cols)} features extracted.")

        return {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "X_train_df": X_train_df,
            "X_test_df": X_test_df,
            "df_train_raw": df_train_full,
            "df_test_raw": df_test_full,
            "feature_names": feature_cols,
            "scaler": self.scaler,
            "G_train": G_train,
            "full_graph": splits["full_graph"],
            "splits": splits
        }

    def transform_single_pair(self, G: nx.Graph, u: int, v: int) -> tuple:
        """
        Extracts features for a single candidate node pair (u, v) in real-time.
        Returns (raw_feature_dict, scaled_feature_vector).
        """
        pair_data = [(u, v, 0)]
        df_topo = compute_edge_features_fast(G, pair_data, self.precomputed_structures)
        
        if self.use_embeddings and self.node2vec_model is not None:
            df_emb = compute_edge_embedding_features(self.node2vec_model.embeddings, pair_data)
            df_full = pd.concat([df_topo, df_emb], axis=1)
        else:
            df_full = df_topo

        # Ensure all expected features exist even if embeddings were disabled
        for col in self.feature_names:
            if col not in df_full.columns:
                df_full[col] = 0.0

        X_df = df_full[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled_vec = self.scaler.transform(X_df.values)
        raw_dict = df_full.iloc[0].to_dict()
        return raw_dict, scaled_vec
