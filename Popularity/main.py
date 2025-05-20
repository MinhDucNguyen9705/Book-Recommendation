#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error
import numpy as np
from model import CBFPopModel
import joblib
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and log enhanced book recommender model with MLflow"
    )
    parser.add_argument(
        "--books_path",
        type=str,
        default="Data/final_books.csv",
        help="Path to books CSV (default: Data/final_books.csv)"
    )
    parser.add_argument(
        "--interactions_path",
        type=str,
        default="Data/interaction.csv",
        help="Path to interactions CSV (default: Data/interaction.csv)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Optional: sample size for interactions (default: None = all rows)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Max features for TF-IDF (default: 5000)"
    )
    parser.add_argument(
        "--w_cbf",
        type=float,
        default=0.3,
        help="Weight for content-based filtering (default: 0.3)"
    )
    parser.add_argument(
        "--w_pop",
        type=float,
        default=0.1,
        help="Weight for popularity (default: 0.1)"
    )
    parser.add_argument(
        "--w_mf",
        type=float,
        default=0.6,
        help="Weight for matrix factorization (default: 0.6)"
    )
    parser.add_argument(
        "--n_factors",
        type=int,
        default=50,
        help="Number of latent factors for matrix factorization (default: 50)"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="Regularization parameter (default: 0.01)"
    )
    parser.add_argument(
        "--normalize_ratings",
        action="store_true",
        help="Enable rating normalization with user and item biases"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set size for validation (default: 0.2)"
    )
    parser.add_argument(
        "--registered_model",
        type=str,
        default=os.getenv("MLFLOW_REGISTERED_MODEL", "ml_catalog.ml_schema.default"),
        help="MLflow registered model name (default: from MLFLOW_REGISTERED_MODEL env)"
    )
    return parser.parse_args()

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "/Users/uytbvn@gmail.com/hybrid_model"))

def load_data(books_path, interactions_path, sample_size=None):
    books = pd.read_csv(books_path)
    interactions = pd.read_csv(interactions_path)
    if sample_size:
        interactions = interactions.sample(n=sample_size, random_state=42)
    return books, interactions

def calculate_rmse(true_vals, pred_vals):
    """Safely calculate RMSE, handling empty arrays"""
    if len(true_vals) == 0:
        return float('inf'), 0
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    return rmse, len(true_vals)

def main():
    args = parse_args()
    setup_mlflow()

    books, interactions = load_data(
        args.books_path, args.interactions_path, args.sample_size
    )
    
    # Split data into train and test sets
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=args.test_size, random_state=42
    )
    
    print(f"Training on {len(train_interactions)} interactions, testing on {len(test_interactions)} interactions")

    with mlflow.start_run() as run:
        # Log all parameters
        mlflow.log_params({
            "max_features": args.max_features,
            "w_cbf": args.w_cbf,
            "w_pop": args.w_pop,
            "w_mf": args.w_mf,
            "n_factors": args.n_factors,
            "regularization": args.regularization,
            "normalize_ratings": args.normalize_ratings,
            "sample_size": args.sample_size if args.sample_size else len(interactions),
            "test_size": args.test_size
        })
        
        # Initialize and train model
        model = CBFPopModel(
            max_features=args.max_features,
            w_cbf=args.w_cbf,
            w_pop=args.w_pop,
            w_mf=args.w_mf,
            n_factors=args.n_factors,
            regularization=args.regularization,
            normalize_ratings=args.normalize_ratings
        )
        
        # Fit on training data
        model.fit(books, train_interactions)
        model.build_prediction_matrix()
        
        # Evaluate on training data
        train_true_vals, train_pred_vals = [], []
        for _, row in train_interactions.iterrows():
            u, b, r = row['user_id'], row['book_id'], row['rating']
            if u in model.pred_df.index and b in model.pred_df.columns:
                train_true_vals.append(r)
                train_pred_vals.append(model.pred_df.at[u, b])
        
        # Safely calculate training RMSE
        train_rmse, train_count = calculate_rmse(train_true_vals, train_pred_vals)
        if train_count > 0:
            mlflow.log_metric("train_rmse", train_rmse)
            print(f"Training RMSE = {train_rmse:.4f} (based on {train_count} predictions)")
        else:
            print("Warning: No overlapping user-item pairs found in training set")
        
        # Evaluate on test data
        test_true_vals, test_pred_vals = [], []
        for _, row in test_interactions.iterrows():
            u, b, r = row['user_id'], row['book_id'], row['rating']
            if u in model.pred_df.index and b in model.pred_df.columns:
                test_true_vals.append(r)
                test_pred_vals.append(model.pred_df.at[u, b])
        
        # Safely calculate test RMSE
        test_rmse, test_count = calculate_rmse(test_true_vals, test_pred_vals)
        if test_count > 0:
            mlflow.log_metric("test_rmse", test_rmse)
            print(f"Test RMSE = {test_rmse:.4f} (based on {test_count} predictions)")
        else:
            print("Warning: No overlapping user-item pairs found in test set")
            print("This can happen if the test users or items don't appear in the training set")
            print("Try increasing the sample size or reducing the test_size")
            # Set a default high RMSE to avoid registration
            test_rmse = float('inf')
        
        # Generate example recommendations
        if len(interactions) > 0:
            example_user = interactions['user_id'].iloc[0]
            input_example = pd.DataFrame({"user_id": [example_user]})
            output_df = model.recommend(example_user, top_n=5)
            output_example = output_df.to_dict(orient="records")
            signature = infer_signature(input_example, output_example)
            
            # Save & log model
            model_path = "enhanced_recommender_model.joblib"
            joblib.dump(model, model_path)
            
            # Only register model if RMSE < 0.9 and we have valid predictions
            if test_count > 0 and test_rmse < 1:
                registered_model_name = args.registered_model
                print(f"RMSE {test_rmse:.4f} < 1, registering model as {registered_model_name}")
            else:
                registered_model_name = None
                if test_count == 0:
                    print("Not registering model due to lack of test predictions")
                else:
                    print(f"RMSE {test_rmse:.4f} >= 0.9, not registering model")
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                code_paths=["Popularity/model.py"]
            )
            os.remove(model_path)
        else:
            print("No interactions available for example recommendations")
        
        print(f"Run complete: {run.info.run_id}")
        if test_count > 0:
            print(f"Final RMSE: {test_rmse:.4f}")
        else:
            print("Final RMSE: Not available (no test predictions)")

if __name__ == "__main__":
    main()
