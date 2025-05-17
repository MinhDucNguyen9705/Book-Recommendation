import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

def load_data():
    X, y = datasets.load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="auto", random_state=8888)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred

def log_model_to_mlflow(model, X_train, acc):
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/uytbvn@gmail.com/MyExperiment")  
    mlflow.login()

    with mlflow.start_run() as run:
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train[:5]
        )

        return run.info.run_id

def load_model_and_predict(run_id, X_test):
    model_uri = f"runs:/{run_id}/iris_model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model.predict(X_test)

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.4f}")

    run_id = log_model_to_mlflow(model, X_train, acc)
    print(f"Model logged under run ID: {run_id}")

    predictions = load_model_and_predict(run_id, X_test)

    df = pd.DataFrame(X_test, columns=datasets.load_iris().feature_names)
    df["actual"] = y_test
    df["predicted"] = predictions
    print(df.head())

if __name__ == "__main__":
    main()
