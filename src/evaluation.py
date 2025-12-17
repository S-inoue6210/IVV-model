import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


import numpy as np
import matplotlib.pyplot as plt

def plot_confidence_margin(df, bins=100):
    """
    Plot Confidence Margin distribution.

    Confidence Margin = proba(top1) - proba(top2)
    If only one candidate exists for a stay_id, proba(top2) is treated as 0.

    Histograms are overlaid for:
      - Correct predictions
      - Incorrect predictions

    df must contain: stay_id, label, proba
    """
    margins_correct = []
    margins_incorrect = []

    for stay_id, group in df.groupby('stay_id'):
        sorted_group = group.sort_values('proba', ascending=False).reset_index(drop=True)

        # Top-1 probability is always available
        proba_1st = sorted_group.loc[0, 'proba']

        # Handle Top-2 probability
        if len(sorted_group) >= 2:
            proba_2nd = sorted_group.loc[1, 'proba']
        # POI候補が1つしかない場合の処理: 要検討（0とすると可視化範囲が広がりすぎる）
        else:
            proba_2nd = 0.0

        margin = proba_1st - proba_2nd

        # Check correctness (Top-1)
        is_correct = sorted_group.loc[0, 'label'] == 1

        if is_correct:
            margins_correct.append(margin)
        else:
            margins_incorrect.append(margin)

    # numpy array化
    margins_correct = np.array(margins_correct)
    margins_incorrect = np.array(margins_incorrect)

    # 共通ビン定義（下限を 0 に固定）
    all_margins = np.concatenate([margins_correct, margins_incorrect])
    max_margin = all_margins.max()

    bin_edges = np.linspace(0, max_margin, bins + 1)

    # plot
    plt.figure(figsize=(8, 6))
    plt.hist(
        margins_correct,
        bins=bin_edges,
        alpha=0.5,
        label='Correct'
    )
    plt.hist(
        margins_incorrect,
        bins=bin_edges,
        alpha=0.5,
        label='Incorrect'
    )

    plt.xlim(0, max_margin)
    plt.xlabel('Confidence Margin (P1 - P2)')
    plt.ylabel('Count')
    plt.title('Confidence Margin Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_entropy(df):
    """
    Calculate entropy of predicted probability distribution per stay_id.

    Returns:
      - entropy_df: stay_id ごとのエントロピー
      - mean_entropy: 全体平均エントロピー

    df must contain: stay_id, proba
    """
    entropy_list = []

    for stay_id, group in df.groupby('stay_id'):
        probs = group['proba'].values

        # numerical stability
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log(probs))
        entropy_list.append({
            'stay_id': stay_id,
            'entropy': entropy
        })

    entropy_df = pd.DataFrame(entropy_list)
    mean_entropy = entropy_df['entropy'].mean()

    return entropy_df, mean_entropy
