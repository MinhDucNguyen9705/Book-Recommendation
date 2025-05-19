import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from yaml import safe_load
from model import HybridModel, save_model
from mlflow.models.signature import infer_signature
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and log book recommender model with MLflow"
    )
    parser.add_argument("--books_path", type=str, default="Data/final_books.csv",
                        help="Path to books CSV; overridden by training_params.yml")
    parser.add_argument("--interactions_path", type=str, default="Data/interaction.csv",
                        help="Path to interactions CSV; overridden by training_params.yml")
    parser.add_argument("--n_factors", type=int, default=20,
                        help="Number of NMF latent factors; overridden by training_params.yml")
    parser.add_argument("--max_iter", type=int, default=200,
                        help="Max iterations for NMF; overridden by training_params.yml")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size for debugging; overridden by training_params.yml if specified")
    return parser.parse_args()


def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "/Users/uytbvn@gmail.com/hybrid_model"))


def load_data(path_b, path_i, sample_size=None):
    df_books = pd.read_csv(path_b)
    df_interactions = pd.read_csv(path_i)
    if sample_size:
        df_books = df_books.sample(n=min(sample_size, len(df_books)), random_state=42)
        df_interactions = df_interactions[df_interactions['book_id'].isin(df_books['book_id'])]
    return df_books, df_interactions


def evaluate_model(model):
    true_vals = model.user_book_matrix.values.flatten()
    pred_vals = model.pred_ratings_df.values.flatten()
    mask = (model.user_book_matrix.values != 0).flatten()  # <-- flatten the mask
    rmse = np.sqrt(mean_squared_error(true_vals[mask], pred_vals[mask]))
    avg_rating = pred_vals[mask].mean()
    return rmse, avg_rating


def create_signature():
    example_input = pd.DataFrame({"user_id": [2142]})
    example_output = pd.Series([{"book1": 4.5, "book2": 3.9}])
    signature = infer_signature(example_input, example_output)
    return signature, example_input


def log_model_to_mlflow(model, model_path, signature, input_example):
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL", "ml_catalog.ml_schema.popularity"),
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )


def main():
    # Read parameters from YAML
    params = safe_load(open('training_params.yml'))
    args = argparse.Namespace(
        books_path=params['books_path'],
        interactions_path=params['interactions_path'],
        n_factors=params['n_factors'],
        max_iter=params['max_iter'],
        sample_size=params.get('sample_size')
    )

    setup_mlflow()
    books, interactions = load_data(
        args.books_path, args.interactions_path, args.sample_size
    )

    with mlflow.start_run() as run:
        # Train
        model = HybridModel(n_factors=args.n_factors, max_iter=args.max_iter)
        model.fit(books, interactions)
        model.predict()

        # Evaluate
        rmse, avg_rating = evaluate_model(model)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("avg_pred_rating", avg_rating)
        mlflow.log_param("n_factors", args.n_factors)
        mlflow.log_param("max_iter", args.max_iter)

        # Signature and example
        signature, input_example = create_signature()

        # Save and log model
        model_path = "book_recommender_model.joblib"
        save_model(model, model_path)
        log_model_to_mlflow(model, model_path, signature, input_example)
        os.remove(model_path)

        print(f"Run complete: {run.info.run_id}")

if __name__ == "__main__":
    main()