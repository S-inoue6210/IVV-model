import numpy as np
import pandas as pd

def calculate_ndcg(df, k_list=[1, 5, 10]):
    """
    Calculate NDCG@k for the dataframe.
    df must contain 'stay_id', 'label', 'score'.
    """
    ndcg_scores = {k: [] for k in k_list}
    
    # Group by stay_id
    grouped = df.groupby('stay_id')
    
    for stay_id, group in grouped:
        # Sort by score descending
        sorted_group = group.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Find rank of the true visit (label=1)
        # Assuming only one true visit per stay_id
        true_visit_indices = sorted_group.index[sorted_group['label'] == 1].tolist()
        
        if not true_visit_indices:
            # No true visit in candidates? Should not happen if data is correct.
            # Skip or assign 0
            for k in k_list:
                ndcg_scores[k].append(0.0)
            continue
            
        true_rank = true_visit_indices[0] + 1 # 1-indexed
        
        for k in k_list:
            if true_rank <= k:
                # DCG = 1 / log2(rank + 1)
                # IDCG = 1 / log2(1 + 1) = 1
                ndcg = (1.0 / np.log2(true_rank + 1))
            else:
                ndcg = 0.0
            ndcg_scores[k].append(ndcg)
            
    # Calculate mean NDCG
    mean_ndcg = {k: np.mean(scores) for k, scores in ndcg_scores.items()}
    return mean_ndcg

def calculate_map(df):
    """
    Calculate Mean Average Precision (MAP).
    """
    ap_scores = []
    
    grouped = df.groupby('stay_id')
    
    for stay_id, group in grouped:
        sorted_group = group.sort_values('score', ascending=False).reset_index(drop=True)
        true_visit_indices = sorted_group.index[sorted_group['label'] == 1].tolist()
        
        if not true_visit_indices:
            ap_scores.append(0.0)
            continue
            
        true_rank = true_visit_indices[0] + 1
        ap = 1.0 / true_rank
        ap_scores.append(ap)
        
    return np.mean(ap_scores)
