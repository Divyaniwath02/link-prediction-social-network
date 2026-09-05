"""
CLI Experiment Runner and Benchmark Suite for Link Prediction in Social Networks.
Runs the complete training and evaluation pipeline, logs performance,
and saves model artifacts for the interactive web dashboard and notebooks.
"""

import os
import json
import time
import pandas as pd
import numpy as np

from src.data_loader import load_graph, split_edges_leak_free
from src.graph_analytics import compute_graph_metrics, get_degree_distribution
from src.pipeline import LinkPredictionPipeline
from src.models import train_model_zoo, save_models_bundle
from src.evaluator import evaluate_all_models, get_feature_importances


def run_pipeline(dataset_name: str = "facebook_ego", sample_nodes: int = 800, use_embeddings: bool = True):
    print("=" * 70)
    print("  LINK PREDICTION IN SOCIAL NETWORKS — EXPERIMENTAL PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Load Graph
    print(f"\n[Step 1/5] Loading Graph Dataset: '{dataset_name}'...")
    G = load_graph(dataset_name=dataset_name, sample_nodes=sample_nodes)
    metrics = compute_graph_metrics(G)
    
    print("\n--- Network Topology Summary ---")
    print(f"Nodes (|V|):                  {metrics['num_nodes']:,}")
    print(f"Edges (|E|):                  {metrics['num_edges']:,}")
    print(f"Density:                      {metrics['density']:.5f}")
    print(f"Average Degree:               {metrics['avg_degree']:.2f}")
    print(f"Average Clustering Coeff:     {metrics['avg_clustering']:.4f}")
    print(f"Network Modularity:           {metrics['modularity']:.4f} ({metrics['num_communities']} communities)")
    print(f"Is Connected:                 {metrics['is_connected']}")
    
    # 2. Leak-Free Graph Splitting
    print("\n[Step 2/5] Partitioning Graph Edges (Leak-Free Spanning Tree Preserving)...")
    splits = split_edges_leak_free(G, test_ratio=0.15, val_ratio=0.05, seed=42)
    meta = splits["metadata"]
    print(f"  Training Edges (G_train):   {meta['train_edges']:,}")
    print(f"  Validation Positive Edges:  {meta['val_positive_edges']:,}")
    print(f"  Test Positive Edges:        {meta['test_positive_edges']:,}")
    print(f"  Training Negative Samples:  {meta['train_negative_edges']:,}")
    print(f"  Test Negative Samples:      {meta['test_negative_edges']:,}")
    print(f"  G_train Connected:          {meta['is_train_connected']}")
    
    # 3. Feature Extraction Pipeline
    print("\n[Step 3/5] Extracting Topological Heuristics and Node2Vec Embeddings...")
    pipeline = LinkPredictionPipeline(use_embeddings=use_embeddings, embedding_dim=16, seed=42)
    data_dict = pipeline.fit_transform_splits(splits)
    print(f"  Extracted Features:         {len(data_dict['feature_names'])}")
    print(f"  Feature List:               {', '.join(data_dict['feature_names'][:8])} ...")
    
    # 4. Model Training
    print("\n[Step 4/5] Training Benchmark Models...")
    models = train_model_zoo(data_dict)
    
    # 5. Model Evaluation
    print("\n[Step 5/5] Evaluating Models on Test Split...")
    leaderboard_df, detailed_metrics = evaluate_all_models(models, data_dict)
    
    print("\n" + "=" * 70)
    print("                      LEADERBOARD SUMMARY")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    print("=" * 70)
    
    # Extract Feature Importances for best tree model (e.g. XGBoost or RF)
    if "XGBoost" in models:
        fi_df = get_feature_importances(models["XGBoost"], data_dict["feature_names"])
        print("\n--- Top 8 Most Predictive Features (XGBoost) ---")
        print(fi_df.head(8).to_string(index=False))
    
    # Save artifacts
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/saved_models", exist_ok=True)
    
    leaderboard_df.to_csv("data/results/benchmark_leaderboard.csv", index=False)
    save_models_bundle(models, pipeline, save_dir="data/saved_models")
    
    elapsed = round(time.time() - start_time, 2)
    print(f"\n[Completed] Entire pipeline finished in {elapsed} seconds!")
    print("Results saved to 'data/results/' and models saved to 'data/saved_models/'")
    return leaderboard_df, models, pipeline, data_dict


if __name__ == "__main__":
    run_pipeline(dataset_name="facebook_ego", sample_nodes=800, use_embeddings=True)
