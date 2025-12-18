import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load data from CSV.
    Expected columns: stay_id, label (0 or 1), feature_01, feature_02, ...
    """
    df = pd.read_csv(filepath)
    return df


def discretize_features(
    df: pd.DataFrame,
    # 離散化対象の {column名, 離散化バケット数}
    discretize_feature: list,
    # 離散化対象外のcolumn名
    non_descretize_feature: list
    ) -> pd.DataFrame:
    """
    Discretize features and return DataFrame with specified column structure.

    discretize_feature: list of dict
      [
        {"feature_name": "distance_bucket", "num_bucket": 40},
        {"feature_name": "rank_bucket", "num_bucket": 40},
        ...
      ]

    non_descretize_feature: list of str
      ["stay_id", "label", ...]

    Output columns:
      - non_descretize_feature
      - discretized feature columns ("feature_name"_bucket)
    """
    df_work = df.copy()

    discretized_columns = []

    for spec in discretize_feature:
        feature = spec["feature_name"]
        num_bucket = spec["num_bucket"]

        if feature not in df_work.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")

        max_val = df_work[feature].max()

        if pd.isna(max_val) or max_val == 0:
            bucket = pd.Series(0, index=df_work.index, dtype="Int64")
        else:
            bucket_width = max_val / num_bucket
            bucket = np.floor(df_work[feature] / bucket_width)
            bucket = bucket.clip(0, num_bucket - 1).astype("Int64")

        bucket_col = f"{feature}_bucket"
        df_work[bucket_col] = bucket
        discretized_columns.append(bucket_col)

    # 出力 column 構成を明示的に指定
    output_columns = non_descretize_feature + discretized_columns

    # 存在チェック（安全装置）
    missing_cols = set(output_columns) - set(df_work.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in output DataFrame: {missing_cols}")

    return df_work[output_columns]


def filter_valid_stays(df):
    """
    Keep only stays where exactly one candidate is labeled 1.
    """
    label_sum = df.groupby('stay_id')['label'].sum()
    valid_stays = label_sum[label_sum == 1].index
    return df[df['stay_id'].isin(valid_stays)].copy()


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
