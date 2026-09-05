# 📽️ Project Presentation Slides: Link Prediction in Social Networks

---

### Slide 1: Title & Overview
**Project Title**: Link Prediction in Social Networks Using Graph Topological Heuristics, Node2Vec, and Supervised Machine Learning  
**Domain**: Network Science, Machine Learning & Social Computing  
**Key Goal**: Predict future / missing friendships between users in a social graph with high accuracy and explainability.

---

### Slide 2: Problem Definition & Sociological Motivation
- **What is Link Prediction?**
  Given a snapshot of a social graph $G = (V, E)$, predict which unconnected pairs $(u, v) \notin E$ will establish a connection in the future.
- **Why is it important?**
  - "People You May Know" recommendations (Facebook, LinkedIn, Instagram).
  - Bioinformatics: Protein-protein interaction network discovery.
  - Cybersecurity: Detecting hidden criminal/fraud rings.
- **Core Challenge**: Extreme Class Imbalance (sparsity of human social ties: $O(V^2)$ non-links vs $O(V)$ real links).

---

### Slide 3: Theoretical Foundations
1. **Triadic Closure**: "A friend of a friend becomes a friend."
2. **Homophily**: People with similar attributes and community roles tend to connect.
3. **Scale-Free Property**: Node degree follows a power-law distribution $P(k) \sim k^{-\gamma}$.

---

### Slide 4: Feature Engineering & Heuristics
- **Local Neighborhood Metrics**:
  - *Common Neighbors*: $|\Gamma(u) \cap \Gamma(v)|$
  - *Jaccard Coefficient*: $\frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}$
  - *Adamic-Adar Index*: $\sum_{z \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log d(z)}$ *(penalizes high-degree hub mutual friends)*
  - *Resource Allocation*: $\sum_{z} \frac{1}{d(z)}$
  - *Preferential Attachment*: $d(u) \cdot d(v)$
- **Representation Learning (Node2Vec)**:
  - Biased 2nd-order random walks parametrized by $p$ (return) and $q$ (in-out exploration).
  - Edge aggregation operators: Cosine similarity, Hadamard product, L1/L2 Euclidean distances.

---

### Slide 5: Methodology — Preventing Data Leakage
- **The Pitfall**: Computing graph features for test pairs on the full graph leads to artificial, inflated accuracy.
- **Our Solution (Leak-Free Spanning Tree Partitioning)**:
  1. Extract Minimum Spanning Tree ($E_{mst}$) to guarantee $G_{train}$ remains fully connected.
  2. Sample test positive edges strictly from non-spanning-tree edges ($E \setminus E_{mst}$).
  3. All topological features and embeddings are computed strictly on $G_{train}$.

---

### Slide 6: Model Zoo & Experimental Results

| Model | ROC-AUC | PR-AUC | Accuracy | F1-Score | Hits@10% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **XGBoost** | **0.9760** | **0.9746** | **92.87%** | **0.9287** | **99.8%** |
| 🥈 **LightGBM** | **0.9759** | 0.9741 | 92.73% | 0.9273 | 99.6% |
| 🥉 **Random Forest** | 0.9754 | 0.9733 | 92.73% | 0.9275 | 99.8% |
| **Spectral GCN** | 0.9737 | 0.9724 | 92.32% | 0.9230 | 100.0% |

**Key Finding**: Resource Allocation (70.4%) and Node2Vec Cosine Similarity (11.3%) are the two strongest signals governing link formation.

---

### Slide 7: Interactive Web Dashboard Demonstration
- **3D Network Sphere**: Interactive 3D force-directed cluster navigation.
- **Top-K Recommender**: "People You May Know" with explainable AI.
- **Connection Heatmap Matrix**: Pairwise link likelihoods across user clusters.
- **Dynamic Decision Threshold Slider**: Real-time confusion matrix updates.
- **What-If Triadic Closure Sandbox**: Live probability jumps when adding mutual friends.

---

### Slide 8: Conclusion & Future Scope
- Combining local topological heuristics with deep representation learning achieves near-optimal link prediction performance (>97.5% AUC).
- Future extensions: Temporal dynamic graph networks (TGCN) and multi-relational knowledge graphs.
