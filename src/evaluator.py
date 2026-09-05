"""
Evaluation and Benchmarking Module for Link Prediction.
Computes ROC-AUC, PR-AUC, F1-Score, Hits@K, Mean Reciprocal Rank (MRR),
generates interactive comparison charts, and calculates feature importances.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve, precision_recall_curve, confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px


def evaluate_single_model(model, X_test, y_test, X_test_df=None, model_name="Model") -> dict:
    """
    Evaluates a single model on test set and returns a comprehensive metrics dictionary.
    """
    # Predict probabilities
    if hasattr(model, "predict_proba"):
        if "Heuristic" in model_name and X_test_df is not None:
            y_probs = model.predict_proba(X_test_df)[:, 1]
        else:
            y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.predict(X_test)

    # Predictions with 0.5 threshold
    y_pred = (y_probs >= 0.5).astype(int)

    # Compute metrics
    roc_auc = float(roc_auc_score(y_test, y_probs))
    pr_auc = float(average_precision_score(y_test, y_probs))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Ranking metric: Hits@K (top 10% predictions)
    k_val = max(10, int(len(y_test) * 0.1))
    top_k_indices = np.argsort(y_probs)[::-1][:k_val]
    hits_at_k = float(np.sum(y_test[top_k_indices]) / min(k_val, np.sum(y_test)))

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "hits_at_10pct": hits_at_k,
        "confusion_matrix": cm,
        "y_probs": y_probs,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals
    }


def evaluate_all_models(models_dict: dict, data_dict: dict) -> tuple:
    """
    Evaluates all trained models on the test set and compiles a benchmark leaderboard.
    """
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    X_test_df = data_dict["X_test_df"]

    results_list = []
    detailed_metrics = {}

    for name, model in models_dict.items():
        metrics = evaluate_single_model(model, X_test, y_test, X_test_df=X_test_df, model_name=name)
        detailed_metrics[name] = metrics
        results_list.append({
            "Model": name,
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "PR-AUC": round(metrics["pr_auc"], 4),
            "F1-Score": round(metrics["f1_score"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "Hits@10%": round(metrics["hits_at_10pct"], 4),
        })

    leaderboard_df = pd.DataFrame(results_list).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return leaderboard_df, detailed_metrics


def plot_roc_curves_plotly(detailed_metrics: dict) -> go.Figure:
    """
    Generates an interactive Plotly ROC curve comparison figure.
    """
    fig = go.Figure()
    
    # Random diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1.5),
        name="Random Chance (AUC = 0.50)"
    ))

    colors = px.colors.qualitative.Plotly

    for i, (name, metrics) in enumerate(detailed_metrics.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=metrics["fpr"],
            y=metrics["tpr"],
            mode="lines",
            line=dict(color=color, width=2.2),
            name=f"{name} (AUC = {metrics['roc_auc']:.3f})"
        ))

    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=50, b=40),
        width=750,
        height=500
    )
    return fig


def plot_pr_curves_plotly(detailed_metrics: dict) -> go.Figure:
    """
    Generates an interactive Plotly Precision-Recall curve comparison figure.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (name, metrics) in enumerate(detailed_metrics.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=metrics["recall_curve"],
            y=metrics["precision_curve"],
            mode="lines",
            line=dict(color=color, width=2.2),
            name=f"{name} (PR-AUC = {metrics['pr_auc']:.3f})"
        ))

    fig.update_layout(
        title="<b>Precision-Recall (PR) Curves</b>",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0.05, y=0.05, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=50, b=40),
        width=750,
        height=500
    )
    return fig


def plot_confusion_matrix_plotly(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> go.Figure:
    """
    Generates an interactive confusion matrix heatmap with real-time thresholding.
    """
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    labels = [["True Neg (Non-Link)", "False Pos (Cheating Link)"],
              ["False Neg (Missed Link)", "True Pos (Predicted Link)"]]
    
    annot_text = [[f"<b>{tn:,}</b><br>TN", f"<b>{fp:,}</b><br>FP"],
                  [f"<b>{fn:,}</b><br>FN", f"<b>{tp:,}</b><br>TP"]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No Link (0)", "Predicted Link (1)"],
        y=["Actual No Link (0)", "Actual Link (1)"],
        text=annot_text,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues",
        showscale=False
    ))
    
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * (prec * rec) / max(1e-9, (prec + rec))
    
    fig.update_layout(
        title=f"<b>Confusion Matrix @ Threshold = {threshold:.2f}</b><br><span style='font-size:12px; color:#94a3b8;'>Acc: {acc*100:.1f}% | Prec: {prec*100:.1f}% | Rec: {rec*100:.1f}% | F1: {f1:.3f}</span>",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        height=320
    )
    return fig


def plot_feature_radar_plotly(raw_feat: dict) -> go.Figure:
    """
    Generates a spider radar chart displaying the pair's multi-dimensional similarity profile.
    """
    categories = ["Common Neighbors", "Adamic-Adar", "Jaccard Index", "Resource Alloc.", "Pref. Attachment", "Node2Vec Sim"]
    
    # Scale values to [0, 1] for relative comparison
    v_cn = min(1.0, raw_feat.get("common_neighbors", 0) / 10.0)
    v_aa = min(1.0, raw_feat.get("adamic_adar", 0.0) / 4.0)
    v_jc = min(1.0, raw_feat.get("jaccard_coefficient", 0.0) * 2.0)
    v_ra = min(1.0, raw_feat.get("resource_allocation", 0.0) * 3.0)
    v_pa = min(1.0, raw_feat.get("preferential_attachment", 0) / 2000.0)
    v_emb = max(0.0, min(1.0, (raw_feat.get("emb_cosine_sim", 0.0) + 1.0) / 2.0))
    
    values = [v_cn, v_aa, v_jc, v_ra, v_pa, v_emb]
    values.append(values[0])
    cat_cycle = categories + [categories[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=cat_cycle,
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.35)",
        line=dict(color="#38bdf8", width=2.5),
        name="Pair Topological Signature"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, color="rgba(255,255,255,0.2)"),
            angularaxis=dict(color="#94a3b8")
        ),
        template="plotly_dark",
        title="<b>Pair Topological Signature (Radar Profile)</b>",
        margin=dict(l=40, r=40, t=40, b=30),
        height=320,
        showlegend=False
    )
    return fig


def plot_3d_network_plotly(G: nx.Graph, max_nodes: int = 80) -> go.Figure:
    """
    Renders an interactive 3D Force-Directed Network Graph in Plotly.
    Users can rotate in 3D, zoom, and explore spatial community clustering.
    """
    import networkx as nx
    sub_nodes = list(G.nodes())[:max_nodes]
    sub_G = G.subgraph(sub_nodes)
    
    # 3D Spring Layout
    pos_3d = nx.spring_layout(sub_G, dim=3, seed=42, k=0.35)
    
    # Extract Edge Coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    for u, v in sub_G.edges():
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.25)", width=1.5),
        hoverinfo="none",
        name="Friendships"
    )
    
    # Extract Node Coordinates
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    degrees = dict(sub_G.degree())
    for node in sub_G.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        deg = degrees[node]
        node_sizes.append(max(5, min(18, 5 + deg * 0.8)))
        node_colors.append(deg)
        node_text.append(f"<b>User {node}</b><br>Degree: {deg} connections")
        
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            opacity=0.9,
            line=dict(width=1, color="white")
        ),
        name="Users"
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="<b>Interactive 3D Social Network Cluster</b> (Drag to rotate in 3D)",
        template="plotly_dark",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showticklabels=False, title=""),
            bgcolor="#0b0f19"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450
    )
    return fig


def plot_link_prediction_matrix_plotly(nodes_subset: list, G: nx.Graph, pipeline, model) -> go.Figure:
    """
    Generates a pairwise Link Probability Heatmap across a community subset.
    Highlights existing links vs highly likely future links!
    """
    n = len(nodes_subset)
    matrix = np.zeros((n, n))
    hover_texts = []
    
    for i, u in enumerate(nodes_subset):
        row_text = []
        for j, v in enumerate(nodes_subset):
            if i == j:
                matrix[i][j] = np.nan
                row_text.append(f"Self User {u}")
            else:
                raw_feat, scaled_vec = pipeline.transform_single_pair(G, u, v)
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(scaled_vec)[0, 1])
                else:
                    p = float(model.predict(scaled_vec)[0])
                matrix[i][j] = p
                status = "<b>ALREADY CONNECTED</b>" if G.has_edge(u, v) else "<b>NEW PREDICTED LINK</b>"
                row_text.append(f"Pair ({u} ↔ {v})<br>Probability: {p*100:.1f}%<br>{status}")
        hover_texts.append(row_text)
        
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"U{u}" for u in nodes_subset],
        y=[f"U{u}" for u in nodes_subset],
        text=hover_texts,
        hoverinfo="text",
        colorscale="Inferno",
        colorbar=dict(title="Probability"),
    ))
    
    fig.update_layout(
        title="<b>Pairwise Future Link Probability Matrix Heatmap</b>",
        xaxis_title="User ID",
        yaxis_title="User ID",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )
    return fig


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """
    Extracts feature importances for Tree models or absolute weights for Linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df

