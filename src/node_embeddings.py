"""
Node Embeddings and Representation Learning for Link Prediction.
Implements Node2Vec (Biased Random Walks + Skip-Gram/Matrix Factorization),
Spectral/Laplacian Graph Embeddings, and edge feature aggregation operators.
"""

import random
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from scipy.sparse.linalg import svds


class Node2Vec:
    """
    Node2Vec: Scalable Feature Learning for Networks (Grover & Leskovec, KDD 2016).
    Learns continuous feature representations for nodes by simulating biased random walks.
    """
    def __init__(self, dimensions: int = 32, walk_length: int = 20, num_walks: int = 10,
                 p: float = 1.0, q: float = 1.0, seed: int = 42):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter: controls likelihood of immediately revisiting node
        self.q = q  # In-out parameter: controls exploration vs local neighborhood (BFS vs DFS)
        self.seed = seed
        self.embeddings = {}

    def _get_alias_edge(self, G: nx.Graph, src: int, dst: int):
        """Computes transition probabilities for second-order random walk step."""
        unnormalized_probs = []
        dst_neighbors = list(G.neighbors(dst))
        
        for dst_nbr in dst_neighbors:
            if dst_nbr == src:
                # Returned back to source node
                prob = 1.0 / self.p
            elif G.has_edge(dst_nbr, src):
                # Distance is 1 (shares neighbor with source)
                prob = 1.0
            else:
                # Distance is 2 (moving outwards)
                prob = 1.0 / self.q
            unnormalized_probs.append(prob)
            
        norm_const = sum(unnormalized_probs)
        if norm_const > 0:
            probs = [p / norm_const for p in unnormalized_probs]
        else:
            probs = [1.0 / len(dst_neighbors)] * len(dst_neighbors)
        return dst_neighbors, probs

    def fit(self, G: nx.Graph):
        """
        Generates biased random walks and fits node embeddings.
        Combines biased walk co-occurrences with low-rank factorization for high speed and stability.
        """
        rng = np.random.default_rng(self.seed)
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        print(f"[Node2Vec] Generating {self.num_walks} walks of length {self.walk_length} (p={self.p}, q={self.q})...")
        
        # Build co-occurrence matrix from walks
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        co_occurrence = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        for walk_iter in range(self.num_walks):
            shuffled_nodes = list(nodes)
            random.Random(self.seed + walk_iter).shuffle(shuffled_nodes)
            for start_node in shuffled_nodes:
                walk = [start_node]
                curr = start_node
                prev = None
                
                for _ in range(self.walk_length - 1):
                    nbrs = list(G.neighbors(curr))
                    if not nbrs:
                        break
                    if prev is None or self.p == self.q == 1.0:
                        # Standard 1st order walk step
                        next_node = random.choice(nbrs)
                    else:
                        cand_nodes, probs = self._get_alias_edge(G, prev, curr)
                        next_node = rng.choice(cand_nodes, p=probs)
                    
                    walk.append(next_node)
                    prev = curr
                    curr = next_node
                
                # Context window updates (window size = 3)
                window = 3
                for i, target in enumerate(walk):
                    target_idx = node_to_idx[target]
                    start_ctx = max(0, i - window)
                    end_ctx = min(len(walk), i + window + 1)
                    for j in range(start_ctx, end_ctx):
                        if i != j:
                            ctx_node = walk[j]
                            ctx_idx = node_to_idx[ctx_node]
                            co_occurrence[target_idx, ctx_idx] += 1.0

        # Positive Pointwise Mutual Information (PPMI) Matrix Factorization
        total_sum = np.sum(co_occurrence)
        if total_sum > 0:
            row_sums = np.sum(co_occurrence, axis=1, keepdims=True) + 1e-9
            col_sums = np.sum(co_occurrence, axis=0, keepdims=True) + 1e-9
            expected = (row_sums @ col_sums) / total_sum
            pmi = np.log(np.maximum(co_occurrence * total_sum / (expected + 1e-9), 1e-9))
            ppmi = np.maximum(pmi, 0.0)
        else:
            ppmi = nx.to_numpy_array(G, nodelist=nodes)

        # Truncated SVD for embedding extraction
        k = min(self.dimensions, n_nodes - 2 if n_nodes > 3 else 1)
        u, s, _ = svds(ppmi, k=k)
        emb_matrix = u * np.sqrt(s)
        
        # Normalize embeddings to unit norm
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms

        self.embeddings = {node: emb_matrix[node_to_idx[node]] for node in nodes}
        print(f"[Node2Vec] Learned embeddings for {len(self.embeddings)} nodes in {self.dimensions} dimensions.")
        return self

    def get_embedding(self, node: int) -> np.ndarray:
        return self.embeddings.get(node, np.zeros(self.dimensions, dtype=np.float32))


class SpectralGraphEmbedding:
    """
    Spectral Graph Embedding via Normalized Graph Laplacian.
    Captures global graph manifold and community clustering structure.
    """
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.embeddings = {}

    def fit(self, G: nx.Graph):
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)
        L_norm = csgraph.laplacian(A, normed=True)
        
        k = min(self.dimensions + 1, n_nodes - 1 if n_nodes > 2 else 1)
        u, s, _ = svds(L_norm, k=k)
        # Drop trivial first eigenvector
        emb = u[:, 1:] if u.shape[1] > 1 else u
        
        # Unit normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        
        self.embeddings = {nodes[i]: emb[i] for i in range(n_nodes)}
        return self

    def get_embedding(self, node: int) -> np.ndarray:
        return self.embeddings.get(node, np.zeros(self.dimensions, dtype=np.float32))


def compute_edge_embedding_features(embeddings_dict: dict, edge_pairs: list, 
                                     operators: list = ["hadamard", "cosine", "l1", "l2"]) -> pd.DataFrame:
    """
    Transforms pairs of node embeddings (z_u, z_v) into edge feature vectors.
    Operators:
    - hadamard: z_u * z_v
    - cosine: (z_u . z_v) / (||z_u|| * ||z_v||)
    - l1: |z_u - z_v|
    - l2: (z_u - z_v)^2
    """
    first_key = next(iter(embeddings_dict.keys()))
    dim = len(embeddings_dict[first_key])
    default_vec = np.zeros(dim, dtype=np.float32)

    records = []
    for pair in edge_pairs:
        u = pair[0]
        v = pair[1]
        
        z_u = embeddings_dict.get(u, default_vec)
        z_v = embeddings_dict.get(v, default_vec)
        
        feat_dict = {}
        
        # Cosine similarity
        norm_u = np.linalg.norm(z_u)
        norm_v = np.linalg.norm(z_v)
        if norm_u > 0 and norm_v > 0:
            cos_sim = float(np.dot(z_u, z_v) / (norm_u * norm_v))
        else:
            cos_sim = 0.0
        feat_dict["emb_cosine_sim"] = cos_sim
        
        # L1 distance sum
        feat_dict["emb_l1_dist"] = float(np.sum(np.abs(z_u - z_v)))
        
        # L2 distance Euclidean
        feat_dict["emb_l2_dist"] = float(np.linalg.norm(z_u - z_v))
        
        # Dot product
        feat_dict["emb_dot_product"] = float(np.dot(z_u, z_v))
        
        # Hadamard components (top dimensions or aggregated stats)
        hadamard = z_u * z_v
        for d in range(min(dim, 8)):
            feat_dict[f"emb_hadamard_{d}"] = float(hadamard[d])
            
        records.append(feat_dict)
        
    return pd.DataFrame(records)
