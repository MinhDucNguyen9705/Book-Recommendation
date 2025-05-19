import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from model import HybridModel, save_model
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and log book recommender model with MLflow"
    )
    parser.add_argument("--books_path", type=str, default="../Data/final_books.csv")
    parser.add_argument("--interactions_path", type=str, default="../Data/interaction.csv")
    parser.add_argument("--n_factors", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=None)
    return parser.parse_args()

def setup_mlflow():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Users/uytbvn@gmail.com/hybrid_model")


def load_data(books_path, interactions_path, sample_size=10000):
    books = pd.read_csv(books_path)
    interactions = pd.read_csv(interactions_path)

    books = books.sample(n=min(sample_size, len(books)), random_state=42)
    interactions = interactions[interactions['book_id'].isin(books['book_id'])]
    return books, interactions


def train_model(books, interactions, n_factors=20):
    model = HybridModel(n_factors=n_factors, max_iter=200)
    model.fit(books, interactions)
    model.predict()
    return model


def evaluate_model(model):
    true_matrix = model.user_book_matrix
    pred_matrix = model.pred_ratings_df

    mask = true_matrix.values != 0
    y_true = true_matrix.values[mask]
    y_pred = pred_matrix.values[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_pred_rating = y_pred.mean()
    return rmse, avg_pred_rating


def create_signature():
    example_input = pd.DataFrame({"user_id": [2142]})
    example_output = pd.Series([{
        "book1": 4.5,
        "book2": 3.9
    }])
    signature = infer_signature(example_input, example_output)
    return signature, example_input


def log_model_to_mlflow(model, model_path, signature, input_example):
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="ml_catalog.ml_schema.popularity",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature
    )

def main():
    args = parse_args()
    setup_mlflow()
    books, interactions = load_data(
        args.books_path, args.interactions_path, args.sample_size
    )
    with mlflow.start_run() as run:
        model = train_model(books, interactions, args.n_factors, args.max_iter)
        rmse, avg_pred_rating = evaluate_model(model)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("avg_pred_rating", avg_pred_rating)
        mlflow.log_param("n_factors", args.n_factors)
        mlflow.log_param("max_iter", args.max_iter)
        signature, input_example = create_signature()
        model_path = "book_recommender_model.joblib"
        save_model(model, model_path)
        log_model_to_mlflow(model, model_path, signature, input_example)
        os.remove(model_path)
        print(f"Run completed. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
