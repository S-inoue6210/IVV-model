import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from CSV.
    Expected columns: stay_id, distance_from_poi, label (0 or 1)
    """
    df = pd.read_csv(filepath)
    return df

def calculate_rank(df):
    """
    Calculate rank of the candidate venue within each stay_id based on distance.
    Rank 0 is the closest venue.
    """
    # Ensure data is sorted by distance just in case, though rank method handles it
    df['rank'] = df.groupby('stay_id')['distance_from_poi'].rank(method='first', ascending=True) - 1
    df['rank'] = df['rank'].astype(int)
    return df

def discretize_features(df, num_buckets=40, max_dist=500):
    """
    Discretize distance and rank into buckets.
    """
    # Discretize Distance
    # 0 to max_dist divided into num_buckets
    # Bucket index = floor(distance / (max_dist / num_buckets))
    # Clip to max bucket index
    
    dist_bin_width = max_dist / num_buckets
    df['distance_bucket'] = np.floor(df['distance_from_poi'] / dist_bin_width).astype(int)
    df['distance_bucket'] = df['distance_bucket'].clip(0, num_buckets - 1)
    
    # Discretize Rank
    # Rank is already an integer. We just clip it to num_buckets - 1
    # Assuming we want one bucket per rank up to a limit?
    # Requirement says: "Rank (0 up to max rank) must be discretized into 40 evenly spaced values (buckets)."
    # If max rank is small, it maps 1-to-1. If large, we might need binning.
    # Usually for rank, we just cap it at 39 (if 0-indexed) or similar.
    # Let's assume simple clipping for now as rank is discrete.
    # Or if we strictly follow "evenly spaced", we might need to know max rank.
    # Given the context, usually top ranks are most important. Let's clip.
    
    df['rank_bucket'] = df['rank'].clip(0, num_buckets - 1)
    
    return df

def train_test_split_by_stay(df, test_ratio=0.2, random_state=42):
    """
    Split data into train and test sets by stay_id.
    """
    stay_ids = df['stay_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(stay_ids)
    
    split_idx = int(len(stay_ids) * (1 - test_ratio))
    train_stay_ids = stay_ids[:split_idx]
    test_stay_ids = stay_ids[split_idx:]
    
    train_df = df[df['stay_id'].isin(train_stay_ids)].copy()
    test_df = df[df['stay_id'].isin(test_stay_ids)].copy()
    
    return train_df, test_df
