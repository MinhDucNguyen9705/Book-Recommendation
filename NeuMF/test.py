import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature
from model import NeuMF, GMF, MLP
from utils import load_data, preprocess_data, split_data
from train import parse_options, train_model
import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    return rmse, y_pred

def log_model_to_mlflow(model, X_train, rmse, args):
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/uytbvn@gmail.com/MyExperiment")  
    mlflow.login()

    with mlflow.start_run() as run:
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("latent_dim", args.latent_dim)
        mlflow.log_param("layers", args.layers)
        mlflow.log_param("regularization", args.regularization)
        mlflow.log_metric("RMSE", rmse)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="neumf_model",
            signature=signature,
            input_example=X_train[:5]
        )

        return run.info.run_id

def load_model_and_predict(run_id, X_test):
    model_uri = f"runs:/{run_id}/neumf_model"
    loaded_model = mlflow.tensorflow.load_model(model_uri)
    return loaded_model.predict(X_test)

def main():
    args = parse_options()

    data, books = load_data('../data/interaction.csv', '../data/final_books.csv')
    data, books, user_encoder, book_encoder, _, genre_columns = preprocess_data(data, books)

    train_df, test_df, train_data, test_data = split_data(data, genre_columns, test_size=0.2)
    train_df, val_df, train_data, val_data = split_data(train_df, genre_columns, test_size=0.1875)

    X_train_user, X_train_book, X_train_genre, y_train = train_data
    X_val_user, X_val_book, X_val_genre, y_val = val_data
    X_test_user, X_test_book, X_test_genre, y_test = test_data

    X_train = [X_train_user, X_train_book, X_train_genre]
    X_val = [X_val_user, X_val_book, X_val_genre]
    X_test = [X_test_user, X_test_book, X_test_genre]

    NUM_USERS = len(data['user_id'].unique())
    NUM_ITEMS = len(data['book_id'].unique())
    NUM_GENRES = len(genre_columns)
    LATENT_DIM = args.latent_dim
    LAYERS = args.layers
    REGULARIZATION = [args.regularization for _ in range (5)]

    model = NeuMF(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, latent_dim=LATENT_DIM, layers_=LAYERS, regs=REGULARIZATION)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
                  loss='mse', 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    train_model(model, train_data, val_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate)
    
    rmse, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error: {rmse:.4f}")

    run_id = log_model_to_mlflow(model, X_train, rmse, args)
    print(f"Model logged under run ID: {run_id}")

    predictions = load_model_and_predict(run_id, X_test)

    df = pd.DataFrame(X_test, columns=['user_id', 'book_id', 'genre'])
    df["user_id"] = user_encoder.inverse_transform(X_test_user.flatten())
    df["book_id"] = book_encoder.inverse_transform(X_test_book.flatten()) 
    df["actual"] = y_test
    df["predicted"] = predictions
    print(df.head())

if __name__ == "__main__":
    main()
