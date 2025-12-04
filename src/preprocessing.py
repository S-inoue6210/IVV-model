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
    df['rank'] = df.groupby('stay_id')['distance_from_poi'].rank(method='first', ascending=True) - 1
    df['rank'] = df['rank'].astype(int)
    return df

def filter_valid_stays(df):
    """
    Keep only stays where exactly one candidate is labeled 1.
    """
    label_sum = df.groupby('stay_id')['label'].sum()
    valid_stays = label_sum[label_sum == 1].index
    return df[df['stay_id'].isin(valid_stays)].copy()

def discretize_features(df, num_buckets=40):
    """
    Discretize distance and rank into buckets using *data-driven max distance*.
    """
    max_dist = df['distance_from_poi'].max()

    if max_dist == 0:
        max_dist = 1e-6

    dist_bin_width = max_dist / num_buckets

    df['distance_bucket'] = np.floor(df['distance_from_poi'] / dist_bin_width).astype(int)
    df['distance_bucket'] = df['distance_bucket'].clip(0, num_buckets - 1)

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
