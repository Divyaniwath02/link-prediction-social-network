"""
Friend Recommendation and Link Explanation Engine for Social Networks.
Provides top-K "People You May Know" recommendations, pair connection analysis,
and explainable graph rationale (triadic closure and mutual friend inspection).
"""

import networkx as nx
import numpy as np
import pandas as pd


def get_candidate_non_neighbors(G: nx.Graph, user_id: int, max_candidates: int = 150) -> list:
    """
    Finds high-priority candidate nodes not currently connected to user_id:
    Prioritizes 2-hop neighbors (friends of friends) followed by influential 3-hop nodes.
    """
    if user_id not in G:
        return []

    direct_neighbors = set(G.neighbors(user_id))
    candidates = set()
    
    # 2-hop neighbors (friends of friends)
    for nbr in direct_neighbors:
        for fof in G.neighbors(nbr):
            if fof != user_id and fof not in direct_neighbors:
                candidates.add(fof)
                if len(candidates) >= max_candidates:
                    break
        if len(candidates) >= max_candidates:
            break

    # If still few candidates, add random non-neighbors
    if len(candidates) < 20:
        all_nodes = list(G.nodes())
        np.random.shuffle(all_nodes)
        for cand in all_nodes:
            if cand != user_id and cand not in direct_neighbors:
                candidates.add(cand)
                if len(candidates) >= max_candidates:
                    break

    return list(candidates)


def recommend_friends_for_user(user_id: int, G: nx.Graph, pipeline, model, top_k: int = 5) -> list:
    """
    Recommends top-K future connections for a given user with explainable reasoning.
    """
    if user_id not in G:
        return []

    candidates = get_candidate_non_neighbors(G, user_id)
    if not candidates:
        return []

    direct_nbrs = set(G.neighbors(user_id))
    results = []

    for cand in candidates:
        cand_nbrs = set(G.neighbors(cand))
        mutual = list(direct_nbrs.intersection(cand_nbrs))
        
        # Extract features
        raw_feat, scaled_vec = pipeline.transform_single_pair(G, user_id, cand)
        
        # Predict probability
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(scaled_vec)[0, 1])
        else:
            prob = float(model.predict(scaled_vec)[0])

        # Explanation logic
        num_mutual = len(mutual)
        if num_mutual > 0:
            explanation = f"Shares {num_mutual} mutual friends: {mutual[:4]}" + (f" and {num_mutual - 4} more" if num_mutual > 4 else "")
        else:
            explanation = "Connected through global community & similar network activity."

        results.append({
            "candidate_id": cand,
            "probability": round(prob, 4),
            "mutual_friends_count": num_mutual,
            "mutual_friends": mutual,
            "common_neighbors": raw_feat.get("common_neighbors", 0),
            "jaccard_coefficient": round(raw_feat.get("jaccard_coefficient", 0.0), 4),
            "adamic_adar": round(raw_feat.get("adamic_adar", 0.0), 4),
            "resource_allocation": round(raw_feat.get("resource_allocation", 0.0), 4),
            "same_community": raw_feat.get("same_community", 0) == 1,
            "explanation": explanation
        })

    # Sort descending by probability
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results[:top_k]


def analyze_user_pair(u: int, v: int, G: nx.Graph, pipeline, model) -> dict:
    """
    Analyzes a specific pair (u, v) and outputs link probability, heuristics, and explanation.
    """
    already_connected = G.has_edge(u, v)
    u_nbrs = set(G.neighbors(u)) if u in G else set()
    v_nbrs = set(G.neighbors(v)) if v in G else set()
    mutual = list(u_nbrs.intersection(v_nbrs))

    raw_feat, scaled_vec = pipeline.transform_single_pair(G, u, v)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled_vec)[0, 1])
    else:
        prob = float(model.predict(scaled_vec)[0])

    is_predicted_link = prob >= 0.5

    return {
        "user_u": u,
        "user_v": v,
        "already_connected": already_connected,
        "predicted_link": is_predicted_link,
        "probability": round(prob, 4),
        "confidence_score": round(prob * 100, 2),
        "mutual_friends_count": len(mutual),
        "mutual_friends": mutual,
        "raw_features": raw_feat,
        "scaled_vector": scaled_vec
    }
