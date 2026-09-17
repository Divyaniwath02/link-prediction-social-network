# 🕸️ Link Prediction in Social Networks

An end-to-end Machine Learning and Graph Analytics system for **Link Prediction in Social Networks** (Project-Based Learning / ML Lab).

Predicts future or hidden connections between users using graph topological heuristics, Node2Vec representation learning, and modern supervised ML/GNN models.

---

## 🌟 Key Features

- **🌐 Real Social Network Datasets**: Stanford SNAP Facebook Ego-network (4,039 nodes, 88k edges), Zachary's Karate Club, and synthetic scale-free networks.
- **🛡️ Leak-Free Graph Splitting**: Spanning-tree-preserving graph partitioning that prevents topological data leakage while maintaining connectivity in $G_{train}$.
- **📐 18+ Topological Heuristics**: Common Neighbors, Jaccard Coefficient, Adamic-Adar Index, Resource Allocation, Preferential Attachment, Sørensen, Salton Cosine, Hub Promoted/Depressed Indices, PageRank diff, Closeness, and Community overlap.
- **🧠 Representation Learning**: Fast Node2Vec biased second-order random walks + Spectral Laplacian embeddings with Hadamard, L1, L2, and Cosine edge operators.
- **🤖 Model Zoo**: LightGBM, XGBoost, Random Forest, Logistic Regression, Multi-Layer Perceptrons (MLP), and Spectral GCN.
- **📊 Comprehensive Evaluation**: ROC-AUC (**0.9760**), PR-AUC (**0.9744**), F1-Score (**0.9290**), Accuracy (**92.89%**), and Hits@10% (**99.8%**).
- **🖥️ Interactive Streamlit Dashboard**: Physics-based 2D/3D network visualizer, "People You May Know" recommendation engine with explainable AI, model arena, and live Triadic Closure "What-If" simulator.
- **📚 Academic Materials**: 4 step-by-step Jupyter Notebooks and a full Project Report ([`PROJECT_REPORT.md`](PROJECT_REPORT.md)) with Viva-Voce Q&A.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments & Train Models (CLI)
```bash
python run_experiments.py
```

### 3. Launch Interactive Web Dashboard (Local)
```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Community Cloud (Free Cloud Hosting)
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Select **Repository**: `Divyaniwath02/link-prediction-social-network`
3. Select **Branch**: `main`
4. Set **Main file path**: `app.py`
5. Click **Deploy!**

---

## 📁 Repository Structure

```
link_prediction/
├── data/
│   ├── raw/                  # Downloaded SNAP Facebook dataset
│   ├── processed/            # Processed feature matrices
│   ├── saved_models/         # Serialized ML models & pipeline bundle
│   └── results/              # Leaderboard CSV & benchmark plots
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset downloader & leak-free edge splitter
│   ├── graph_analytics.py    # Power law fit, degree distribution & centrality
│   ├── heuristics.py         # Vectorized topological similarity heuristics
│   ├── node_embeddings.py    # Node2Vec & Spectral graph embeddings
│   ├── pipeline.py           # Feature engineering & dataset builder
│   ├── models.py             # LightGBM, XGBoost, RF, LR, MLP, GCN
│   ├── evaluator.py          # ROC-AUC, PR-AUC, Hits@K, interactive curves
│   └── recommender.py        # Explainable friend recommendation engine
├── notebooks/
│   ├── 01_EDA_and_Graph_Analysis.ipynb
│   ├── 02_Heuristics_and_Feature_Engineering.ipynb
│   ├── 03_ML_Model_Training_and_Comparison.ipynb
│   └── 04_Full_PBL_Demo_and_Inference.ipynb
├── app.py                    # Streamlit interactive application
├── run_experiments.py        # CLI benchmark execution runner
├── PROJECT_REPORT.md         # Comprehensive PBL Academic Report
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Benchmark Leaderboard

| Model | ROC-AUC | PR-AUC | F1-Score | Accuracy | Hits@10% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** | **0.9760** | **0.9744** | **0.9290** | **92.89%** | 99.59% |
| 🥈 **XGBoost** | **0.9759** | **0.9743** | **0.9283** | **92.83%** | **99.80%** |
| 🥉 **Random Forest** | 0.9754 | 0.9734 | 0.9275 | 92.73% | **99.80%** |
| **Logistic Regression** | 0.9744 | 0.9734 | 0.9234 | 92.37% | **100.00%** |
| **Multi-Layer Perceptron** | 0.9740 | 0.9723 | 0.9235 | 92.35% | 99.80% |
| **Spectral GCN** | 0.9737 | 0.9724 | 0.9228 | 92.30% | **100.00%** |
| *Heuristic (Resource Allocation)* | 0.9727 | 0.9702 | 0.0000 | 50.00% | 99.80% |
| *Heuristic (Adamic-Adar)* | 0.9679 | 0.9629 | 0.0048 | 50.12% | 99.39% |

---

## 📖 Theoretical Highlights
- **Triadic Closure**: Pairs with higher numbers of mutual friends ($A-B-C$) have high statistical propensity to form connections ($A-C$).
- **Adamic-Adar vs Common Neighbors**: Adamic-Adar weights mutual friends by $\frac{1}{\log d(z)}$, heavily discounting common celebrity hubs.
- **Node2Vec**: Flexible biased random walks that capture both local homophily ($p$) and global structural roles ($q$).
