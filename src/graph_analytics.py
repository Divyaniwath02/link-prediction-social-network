"""
Graph Analytics and Network Exploratory Data Analysis (EDA) Module.
Computes comprehensive topological statistics, degree distributions, centrality metrics,
and community structure for social networks.
"""

import math
import networkx as nx
import numpy as np
import pandas as pd


def compute_graph_metrics(G: nx.Graph, sample_path_length: bool = True) -> dict:
    """
    Computes global topological metrics of the social network graph.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes == 0 or n_edges == 0:
        return {"num_nodes": n_nodes, "num_edges": n_edges}

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees))
    max_degree = int(np.max(degrees))
    min_degree = int(np.min(degrees))
    density = nx.density(G)
    
    # Clustering
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    # Connected components
    n_components = nx.number_connected_components(G)
    is_conn = nx.is_connected(G)
    
    # Path length and diameter
    if is_conn:
        if n_nodes <= 1500 or not sample_path_length:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # Sample approximation for large graphs
            sample_nodes = np.random.choice(list(G.nodes()), size=min(150, n_nodes), replace=False)
            lengths = []
            max_sampled_len = 0
            for src in sample_nodes:
                spl = nx.single_source_shortest_path_length(G, src)
                for dst, l in spl.items():
                    if src != dst:
                        lengths.append(l)
                        if l > max_sampled_len:
                            max_sampled_len = l
            avg_path_length = float(np.mean(lengths)) if lengths else 0.0
            diameter = max_sampled_len
    else:
        avg_path_length = float("nan")
        diameter = float("nan")

    # Community detection
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        modularity = nx.community.modularity(G, communities)
        num_communities = len(communities)
    except Exception:
        modularity = 0.0
        num_communities = 1

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": density,
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "num_components": n_components,
        "is_connected": is_conn,
        "avg_path_length": avg_path_length,
        "diameter": diameter,
        "modularity": modularity,
        "num_communities": num_communities,
    }


def get_degree_distribution(G: nx.Graph) -> dict:
    """
    Calculates degree distribution and estimates power-law exponent gamma.
    """
    degrees = [d for _, d in G.degree() if d > 0]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    probabilities = counts / len(degrees)
    
    # Power-law estimation: log(P(k)) = -gamma * log(k) + c
    log_k = np.log(unique_degrees)
    log_p = np.log(probabilities)
    
    if len(log_k) > 2:
        # Linear regression on log-log
        slope, intercept = np.polyfit(log_k, log_p, 1)
        gamma = -slope
        r_squared = float(np.corrcoef(log_k, log_p)[0, 1] ** 2)
    else:
        gamma = 0.0
        r_squared = 0.0

    return {
        "degrees": unique_degrees.tolist(),
        "counts": counts.tolist(),
        "probabilities": probabilities.tolist(),
        "power_law_gamma": float(gamma),
        "r_squared": float(r_squared),
    }


def compute_node_centralities(G: nx.Graph, top_k: int = 10) -> pd.DataFrame:
    """
    Computes top node centrality rankings across Degree, PageRank, Closeness, and Betweenness.
    """
    deg_cent = nx.degree_centrality(G)
    pagerank_cent = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # For large graphs, approximate betweenness/closeness if needed
    if G.number_of_nodes() <= 1000:
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
    else:
        betweenness_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        closeness_cent = nx.closeness_centrality(G)

    df = pd.DataFrame({
        "node_id": list(G.nodes()),
        "degree": [G.degree(n) for n in G.nodes()],
        "degree_centrality": [deg_cent[n] for n in G.nodes()],
        "pagerank": [pagerank_cent[n] for n in G.nodes()],
        "betweenness_centrality": [betweenness_cent.get(n, 0.0) for n in G.nodes()],
        "closeness_centrality": [closeness_cent.get(n, 0.0) for n in G.nodes()],
    })

    return df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
