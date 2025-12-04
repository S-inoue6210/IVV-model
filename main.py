import pandas as pd
import numpy as np

from src.preprocessing import load_data, calculate_rank, filter_valid_stays, discretize_features, train_test_split_by_stay
from src.model import IVVModel
from src.evaluation import calculate_ndcg, calculate_map

def main():
    # 1. Load Data
    print("Loading data...")
    filepath = "data/data_01.csv"
    try:
        df = load_data(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return

    print(f"Data loaded: {len(df)} rows.")

    # 2. Preprocessing
    print("Preprocessing...")
    if 'rank' not in df.columns:
        print("Calculating rank...")
        df = calculate_rank(df)
    
    print("Filtering valid stays...")
    df = filter_valid_stays(df)
    print(f"Valid stays: {len(df)} rows.")

    print("Discretizing features...")
    df = discretize_features(df)
    
    # 3. Train/Test Split
    print("Splitting data...")
    train_df, test_df = train_test_split_by_stay(df, test_ratio=0.2)
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    
    # 4. Model Training
    print("Training model...")
    model = IVVModel()
    model.train(train_df, learning_rate=0.001, iterations=100, verbose=True)
    
    # 5. Evaluation
    print("Evaluating on Test Set...")
    # Predict scores for test set
    test_df_pred = model.predict_proba(test_df)
    
    # Calculate Metrics
    ndcg = calculate_ndcg(test_df_pred, k_list=[1, 5, 10])
    map_score = calculate_map(test_df_pred)
    
    print("\nEvaluation Results:")
    print(f"MAP: {map_score:.4f}")
    for k, score in ndcg.items():
        print(f"NDCG@{k}: {score:.4f}")
        
    # Also evaluate on Train set for sanity check
    print("\n(Sanity Check) Train Set Metrics:")
    train_df_pred = model.predict_proba(train_df)
    train_map = calculate_map(train_df_pred)
    print(f"Train MAP: {train_map:.4f}")

if __name__ == "__main__":
    main()
