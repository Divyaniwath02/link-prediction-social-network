"""
Streamlit Web Application: Link Prediction in Social Networks
Interactive 2D/3D Graph Visualization, Friend Recommendation Engine,
Model Performance Arena, Pairwise Connection Matrix, and What-If Simulation Sandbox.
"""

import os
import io
import math
import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix

from src.data_loader import load_graph, split_edges_leak_free
from src.graph_analytics import compute_graph_metrics, get_degree_distribution, compute_node_centralities
from src.heuristics import precompute_graph_structures
from src.pipeline import LinkPredictionPipeline
from src.models import train_model_zoo, save_models_bundle, load_models_bundle
from src.evaluator import evaluate_all_models, get_feature_importances
from src.recommender import recommend_friends_for_user, analyze_user_pair

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="Link Prediction in Social Networks | ML PBL",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------- VISUALIZATION & HELPER FUNCTIONS ----------------

def load_custom_graph(file_obj) -> nx.Graph:
    """Loads a custom graph from an uploaded CSV or TXT edge list file."""
    try:
        if isinstance(file_obj, bytes):
            content = file_obj.decode("utf-8")
        elif hasattr(file_obj, "read"):
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
        else:
            content = str(file_obj)

        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith(("#", "%"))]
        edges = []
        for line in lines:
            parts = [p.strip() for p in line.replace(",", " ").replace(";", " ").replace("\t", " ").split()]
            if len(parts) >= 2:
                if not parts[0].isdigit() or not parts[1].isdigit():
                    continue
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edges.append((u, v))

        G = nx.Graph()
        G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        if not nx.is_connected(G) and G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, mapping)
        G.name = "Custom Uploaded Graph"
        return G
    except Exception:
        return nx.karate_club_graph()


def plot_roc_curves_plotly(detailed_metrics: dict) -> go.Figure:
    fig = go.Figure()
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
        height=450
    )
    return fig


def plot_pr_curves_plotly(detailed_metrics: dict) -> go.Figure:
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
        height=450
    )
    return fig


def plot_confusion_matrix_plotly(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> go.Figure:
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
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
    categories = ["Common Neighbors", "Adamic-Adar", "Jaccard Index", "Resource Alloc.", "Pref. Attachment", "Node2Vec Sim"]
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
    sub_nodes = list(G.nodes())[:max_nodes]
    sub_G = G.subgraph(sub_nodes)
    pos_3d = nx.spring_layout(sub_G, dim=3, seed=42, k=0.35)
    
    edge_x, edge_y, edge_z = [], [], []
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
    
    node_x, node_y, node_z, node_text, node_colors, node_sizes = [], [], [], [], [], []
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


# ---------------- CUSTOM STYLES ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-pos {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neg {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_or_train_pipeline(dataset_name: str = "facebook_ego", sample_nodes: int = 800, use_embeddings: bool = True, custom_bytes=None):
    if custom_bytes is not None:
        G = load_custom_graph(custom_bytes)
    else:
        bundle_path = os.path.join(BASE_DIR, "data", "saved_models", "link_prediction_bundle.joblib")
        if os.path.exists(bundle_path) and dataset_name == "facebook_ego" and sample_nodes == 800:
            try:
                models, pipeline = load_models_bundle(bundle_path)
                G = load_graph(dataset_name=dataset_name, sample_nodes=sample_nodes)
                splits = split_edges_leak_free(G, test_ratio=0.15, val_ratio=0.05, seed=42)
                data_dict = pipeline.fit_transform_splits(splits)
                return G, splits, pipeline, models, data_dict
            except Exception:
                pass
        G = load_graph(dataset_name=dataset_name, sample_nodes=sample_nodes)

    splits = split_edges_leak_free(G, test_ratio=0.15, val_ratio=0.05, seed=42)
    pipeline = LinkPredictionPipeline(use_embeddings=use_embeddings, embedding_dim=16, seed=42)
    data_dict = pipeline.fit_transform_splits(splits)
    models = train_model_zoo(data_dict)
    if custom_bytes is None:
        save_models_bundle(models, pipeline)
    return G, splits, pipeline, models, data_dict


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/share.png", width=56)
    st.markdown("### ⚙️ Graph Data Source")
    
    dataset_option = st.selectbox(
        "Select Dataset",
        options=["facebook_ego", "facebook", "karate", "synthetic_social", "custom_upload"],
        format_func=lambda x: {
            "facebook_ego": "Facebook Ego-Network (~800 nodes)",
            "facebook": "Full SNAP Facebook (4,039 nodes)",
            "karate": "Zachary's Karate Club (34 nodes)",
            "synthetic_social": "Synthetic Scale-Free Network",
            "custom_upload": "📁 Upload Custom Edge List (CSV/TXT)"
        }[x],
        index=0
    )
    
    custom_bytes = None
    if dataset_option == "custom_upload":
        uploaded_file = st.file_uploader("Upload CSV/TXT edge list (source, target)", type=["txt", "csv"])
        if uploaded_file is not None:
            custom_bytes = uploaded_file.read()
        else:
            st.info("Upload an edge list or switch back to standard datasets.")
            dataset_option = "facebook_ego"

    sample_nodes_val = 800 if dataset_option == "facebook_ego" else None
    
    st.markdown("---")
    st.markdown("### 🧠 Active Prediction Model")
    selected_model_name = st.selectbox(
        "Model for Live Inference",
        options=["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "Multi-Layer Perceptron (MLP)", "Spectral GCN", "Heuristic (adamic_adar)", "Heuristic (resource_allocation)"],
        index=0
    )
    
    st.markdown("---")
    retrain_btn = st.button("🔄 Retrain / Refresh Models", width="stretch")
    if retrain_btn:
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 15px;">
    <b>PBL Project:</b> Link Prediction in Social Networks<br>
    <b>Techniques:</b> Graph Heuristics, Node2Vec, GBDT, GCN, Triadic Closure
    </div>
    """, unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
with st.spinner("Initializing Social Graph and Machine Learning Models..."):
    G, splits, pipeline, models, data_dict = get_or_train_pipeline(
        dataset_name=dataset_option,
        sample_nodes=sample_nodes_val,
        use_embeddings=True,
        custom_bytes=custom_bytes
    )
    active_model = models.get(selected_model_name, models["XGBoost"])
    graph_stats = compute_graph_metrics(G)


# ---------------- MAIN HEADER ----------------
st.markdown('<div class="main-header">Link Prediction in Social Networks</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict future relationships, uncover hidden ties, and explore network topology dynamics with Machine Learning & Representation Learning.</div>', unsafe_allow_html=True)

# ---------------- NAVIGATION TABS ----------------
tabs = st.tabs([
    "📊 Network Intelligence & 3D Explorer",
    "🕸️ 2D Physics Visualizer",
    "🤝 Friend Recommender & Pair Profiler",
    "🗺️ Connection Heatmap Matrix",
    "🏆 Model Performance Arena",
    "🧪 What-If Simulation Sandbox"
])


# ==============================================================================
# TAB 1: NETWORK INTELLIGENCE & 3D EXPLORER
# ==============================================================================
with tabs[0]:
    st.markdown("### 🌐 Global Graph Topology Metrics")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Users (|V|)</div>
            <div class="metric-val">{graph_stats['num_nodes']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Friendships (|E|)</div>
            <div class="metric-val">{graph_stats['num_edges']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Network Density</div>
            <div class="metric-val">{graph_stats['density']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Avg Clustering Coeff</div>
            <div class="metric-val">{graph_stats['avg_clustering']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📈 Degree Distribution & Power-Law Fit")
        deg_data = get_degree_distribution(G)
        
        fig_deg = go.Figure()
        fig_deg.add_trace(go.Scatter(
            x=deg_data["degrees"],
            y=deg_data["probabilities"],
            mode="markers",
            marker=dict(size=8, color="#38bdf8", opacity=0.8),
            name="Empirical P(k)"
        ))
        
        gamma = deg_data["power_law_gamma"]
        if gamma > 0:
            k_arr = np.array(deg_data["degrees"])
            c_fit = deg_data["probabilities"][0] * (k_arr[0] ** gamma)
            p_fit = c_fit * (k_arr ** (-gamma))
            fig_deg.add_trace(go.Scatter(
                x=k_arr, y=p_fit,
                mode="lines",
                line=dict(color="#f43f5e", dash="dash", width=2),
                name=f"Power-Law Fit (γ = {gamma:.2f}, R² = {deg_data['r_squared']:.2f})"
            ))

        fig_deg.update_layout(
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="Degree (k) [Log scale]",
            yaxis_title="Probability P(k) [Log scale]",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=30, b=40),
            height=340
        )
        st.plotly_chart(fig_deg, width="stretch")
        st.caption("✨ Demonstrates the **Scale-Free property** ($P(k) \\sim k^{-\\gamma}$) characteristic of human social graphs.")

    with col_right:
        st.markdown("#### 🌟 3D Interactive Force-Directed Network")
        fig_3d = plot_3d_network_plotly(G, max_nodes=70)
        st.plotly_chart(fig_3d, width="stretch")
        st.caption("💡 Rotate, spin, and zoom in 3D to explore spatial communities and node density.")


# ==============================================================================
# TAB 2: 2D PHYSICS NETWORK VISUALIZER
# ==============================================================================
with tabs[1]:
    st.markdown("### 🕸️ Physics-Driven 2D Network Explorer (PyVis)")
    st.markdown("Drag and interact with users and their friendship connections.")
    
    col_v1, col_v2 = st.columns([1, 3])
    with col_v1:
        subgraph_user = st.selectbox(
            "Focus User Ego-Network",
            options=sorted(list(G.nodes()))[:100],
            index=0
        )
        hop_radius = st.slider("Neighborhood Radius (Hops)", min_value=1, max_value=2, value=1)
        max_subgraph_nodes = st.slider("Max Display Nodes", min_value=15, max_value=80, value=40)
        
    with col_v2:
        ego_nodes = set(nx.single_source_shortest_path_length(G, subgraph_user, cutoff=hop_radius).keys())
        if len(ego_nodes) > max_subgraph_nodes:
            ego_nodes = list(ego_nodes)[:max_subgraph_nodes]
            if subgraph_user not in ego_nodes:
                ego_nodes.append(subgraph_user)
        
        sub_G = G.subgraph(ego_nodes).copy()
        comm_map = pipeline.precomputed_structures.get("community_map", {})
        
        net = Network(height="480px", width="100%", bgcolor="#0f172a", font_color="#e2e8f0", cdn_resources="remote")
        net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=90, spring_strength=0.08)
        
        community_colors = ["#38bdf8", "#818cf8", "#c084fc", "#f43f5e", "#fb923c", "#4ade80", "#e879f9", "#22d3ee"]
        
        for node in sub_G.nodes():
            is_focus = (node == subgraph_user)
            comm_id = comm_map.get(node, 0)
            color = "#f43f5e" if is_focus else community_colors[comm_id % len(community_colors)]
            size = 24 if is_focus else (12 + min(sub_G.degree(node) * 1.5, 20))
            label = f"User {node} (FOCUS)" if is_focus else f"User {node}"
            title = f"User {node}<br>Degree: {G.degree(node)}<br>Community: {comm_id}"
            net.add_node(node, label=label, color=color, size=size, title=title)
            
        for u, v in sub_G.edges():
            net.add_edge(u, v, color="rgba(148, 163, 184, 0.4)", width=1.5)
            
        html_content = net.generate_html()
        components.html(html_content, height=500, scrolling=False)
        st.caption(f"Viewing User **{subgraph_user}** ego-network with {sub_G.number_of_nodes()} users and {sub_G.number_of_edges()} connections.")


# ==============================================================================
# TAB 3: FRIEND RECOMMENDER & PAIR PROFILER
# ==============================================================================
with tabs[2]:
    mode_selection = st.radio("Select Prediction Mode:", ["👤 People You May Know (Top-K Recommendation)", "🔍 Specific Pair Link Predictor & Radar Profile"], horizontal=True)
    
    if mode_selection == "👤 People You May Know (Top-K Recommendation)":
        st.markdown("#### 🤝 Top \"People You May Know\" Recommendations")
        col_rec1, col_rec2 = st.columns([1, 2])
        
        with col_rec1:
            rec_user = st.selectbox("Select User ID:", options=sorted(list(G.nodes()))[:100], index=0)
            top_k_val = st.slider("Top Recommendations (K)", min_value=3, max_value=10, value=5)
            st.info(f"**Target User:** `User {rec_user}`\n- Existing Friends: {G.degree(rec_user)}\n- Model: `{selected_model_name}`")
            
        with col_rec2:
            recs = recommend_friends_for_user(rec_user, G, pipeline, active_model, top_k=top_k_val)
            if not recs:
                st.warning("No high-confidence candidates found.")
            else:
                for rank, rec in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">#{rank} Recommend User {rec['candidate_id']}</span>
                                <span class="badge-pos" style="margin-left: 10px;">{rec['probability']*100:.1f}% Probability</span>
                            </div>
                            <span style="color: #94a3b8; font-size: 0.9rem;">Adamic-Adar: <b>{rec['adamic_adar']}</b></span>
                        </div>
                        <div style="margin-top: 8px; color: #cbd5e1; font-size: 0.92rem;">
                            💡 <b>Explanation:</b> {rec['explanation']}
                        </div>
                        <div style="margin-top: 6px; font-size: 0.82rem; color: #64748b;">
                            Jaccard: {rec['jaccard_coefficient']} | Resource Allocation: {rec['resource_allocation']} | Same Community: {rec['same_community']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("#### 🔍 Pair Link Predictor & Similarity Radar Profile")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            user_u = st.selectbox("Select User A:", options=sorted(list(G.nodes()))[:100], index=0)
        with c_p2:
            user_v = st.selectbox("Select User B:", options=sorted(list(G.nodes()))[:100], index=1)
            
        if user_u == user_v:
            st.warning("Please select two distinct users.")
        else:
            analysis = analyze_user_pair(user_u, user_v, G, pipeline, active_model)
            prob_pct = analysis["confidence_score"]
            
            c_g, c_rad = st.columns([1, 1])
            with c_g:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    title={'text': "<b>Link Probability</b>", 'font': {'size': 20, 'color': '#f8fafc'}},
                    number={'suffix': "%", 'font': {'size': 36, 'color': '#38bdf8'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#38bdf8" if prob_pct >= 50 else "#f43f5e"},
                        'bgcolor': "#1e293b",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                            {'range': [50, 80], 'color': 'rgba(251, 146, 60, 0.15)'},
                            {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, width="stretch")

                st.markdown(f"**Status:** {'<span class=\"badge-pos\">CONNECTED</span>' if analysis['already_connected'] else '<span class=\"badge-neg\">UNCONNECTED</span>'} | **Prediction:** {'<span class=\"badge-pos\">LIKELY TO CONNECT</span>' if analysis['predicted_link'] else '<span class=\"badge-neg\">UNLIKELY</span>'}", unsafe_allow_html=True)
                st.markdown(f"- **Mutual Friends:** {analysis['mutual_friends_count']} {analysis['mutual_friends'][:6]}")

            with c_rad:
                fig_rad = plot_feature_radar_plotly(analysis["raw_features"])
                st.plotly_chart(fig_rad, width="stretch")


# ==============================================================================
# TAB 4: CONNECTION HEATMAP MATRIX
# ==============================================================================
with tabs[3]:
    st.markdown("### 🗺️ Pairwise Connection Probability Matrix Heatmap")
    st.markdown("Explore pairwise connection likelihoods across an entire user cluster.")
    
    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        cluster_start = st.slider("User Subset Start Index", min_value=0, max_value=max(0, G.number_of_nodes() - 15), value=0)
        cluster_size = st.slider("Subset Size", min_value=6, max_value=16, value=10)
        user_subset = list(sorted(G.nodes()))[cluster_start : cluster_start + cluster_size]
    with col_m2:
        fig_mat = plot_link_prediction_matrix_plotly(user_subset, G, pipeline, active_model)
        st.plotly_chart(fig_mat, width="stretch")
        st.caption("🔥 Darker red/gold cells indicate high probability of establishing a future social tie.")


# ==============================================================================
# TAB 5: MODEL PERFORMANCE ARENA
# ==============================================================================
with tabs[4]:
    st.markdown("### 🏆 Comprehensive Model Leaderboard & Evaluation Arena")
    
    leaderboard_df, detailed_metrics = evaluate_all_models(models, data_dict)
    
    st.dataframe(
        leaderboard_df.style.highlight_max(axis=0, color="#1e3a8a"),
        width="stretch",
        hide_index=True
    )
    
    col_roc, col_pr = st.columns(2)
    with col_roc:
        fig_roc = plot_roc_curves_plotly(detailed_metrics)
        st.plotly_chart(fig_roc, width="stretch")
        
    with col_pr:
        fig_pr = plot_pr_curves_plotly(detailed_metrics)
        st.plotly_chart(fig_pr, width="stretch")
        
    st.markdown("---")
    st.markdown("#### 🎛️ Real-Time Dynamic Decision Threshold Slider")
    thresh_val = st.slider("Classification Probability Threshold:", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    
    col_cm, col_fi = st.columns([1, 1])
    with col_cm:
        active_metric_data = detailed_metrics.get(selected_model_name, detailed_metrics["XGBoost"])
        fig_cm = plot_confusion_matrix_plotly(data_dict["y_test"], active_metric_data["y_probs"], threshold=thresh_val)
        st.plotly_chart(fig_cm, width="stretch")
        
    with col_fi:
        if "XGBoost" in models:
            fi_df = get_feature_importances(models["XGBoost"], data_dict["feature_names"]).head(8)
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=40, r=40, t=30, b=40))
            st.plotly_chart(fig_fi, width="stretch")


# ==============================================================================
# TAB 6: WHAT-IF SIMULATION SANDBOX
# ==============================================================================
with tabs[5]:
    st.markdown("### 🧪 What-If Network Simulation: Triadic Closure Demonstration")
    st.markdown(r"""
    **Triadic Closure Principle**: In social networks, if user $A$ is friends with $B$, and user $B$ is friends with $C$, 
    there is a strong organic tendency for $A$ and $C$ to form a friendship ($A-B-C \implies A-C$).
    
    *Test this live:* Pick two unconnected users and add hypothetical mutual friends between them to watch the predicted probability rise in real-time!
    """)
    
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        sim_u = st.selectbox("User 1:", options=sorted(list(G.nodes()))[:60], index=5)
    with c_s2:
        sim_v = st.selectbox("User 2:", options=sorted(list(G.nodes()))[:60], index=15)
        
    if sim_u != sim_v:
        base_analysis = analyze_user_pair(sim_u, sim_v, G, pipeline, active_model)
        
        st.markdown("---")
        st.markdown("#### ⚡ Dynamic Intervention")
        num_new_mutuals = st.slider("Add Hypothetical Mutual Friends between User 1 and User 2:", min_value=0, max_value=8, value=0)
        
        sim_G = G.copy()
        if num_new_mutuals > 0:
            existing_max_node = max(sim_G.nodes())
            for i in range(num_new_mutuals):
                fake_node = existing_max_node + 1000 + i
                sim_G.add_node(fake_node)
                sim_G.add_edge(sim_u, fake_node)
                sim_G.add_edge(sim_v, fake_node)
                
        sim_pipeline = LinkPredictionPipeline(use_embeddings=pipeline.use_embeddings)
        sim_pipeline.precomputed_structures = precompute_graph_structures(sim_G)
        sim_pipeline.scaler = pipeline.scaler
        sim_pipeline.feature_names = pipeline.feature_names
        sim_pipeline.node2vec_model = pipeline.node2vec_model
        
        sim_analysis = analyze_user_pair(sim_u, sim_v, sim_G, sim_pipeline, active_model)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Initial Link Probability",
                value=f"{base_analysis['confidence_score']:.1f}%",
                delta=None
            )
        with m2:
            delta_prob = sim_analysis['confidence_score'] - base_analysis['confidence_score']
            st.metric(
                label=f"Simulated Probability (+{num_new_mutuals} mutual friends)",
                value=f"{sim_analysis['confidence_score']:.1f}%",
                delta=f"+{delta_prob:.1f}%" if delta_prob > 0 else f"{delta_prob:.1f}%"
            )
        with m3:
            st.metric(
                label="Adamic-Adar Score Shift",
                value=f"{sim_analysis['raw_features'].get('adamic_adar', 0):.2f}",
                delta=f"+{sim_analysis['raw_features'].get('adamic_adar', 0) - base_analysis['raw_features'].get('adamic_adar', 0):.2f}"
            )
            
        st.success(f"🎯 **Empirical Confirmation:** Adding {num_new_mutuals} shared friend(s) shifted link probability from **{base_analysis['confidence_score']:.1f}%** to **{sim_analysis['confidence_score']:.1f}%**.")
