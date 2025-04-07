from src.data.preprocess import load_and_combine
from src.features.vectorizer import vectorize_text, save_vectorizer
from src.models.train import train_models, save_models
from src.models.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
import mlflow
import mlflow.sklearn
import os

# --- Paths ---
true_path = "D:/IIT Madras/SEM 2/ML OPS LAB/End-to-End-MLOPS-Project/dataset/True.csv"
fake_path = "D:/IIT Madras/SEM 2/ML OPS LAB/End-to-End-MLOPS-Project/dataset/Fake.csv"
model_dir = "D:/IIT Madras/SEM 2/ML OPS LAB/End-to-End-MLOPS-Project/models"
os.makedirs(model_dir, exist_ok=True)

# --- MLflow setup ---
mlflow.set_tracking_uri(uri="http://127.0.0.1:8082")
mlflow.set_experiment("Fake-News-Detection")

with mlflow.start_run(run_name="Fake News Training Session"):

    # Load data and vectorize
    df = load_and_combine(true_path, fake_path)
    X, tfidf = vectorize_text(df)

    # Log vectorizer info
    mlflow.log_param("tfidf_max_features", tfidf.max_features if hasattr(tfidf, "max_features") else "default")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

    # Train models
    models = train_models(X_train, y_train)

    # Save vectorizer once
    vectorizer_path = f"{model_dir}/tfidf_vectorizer.pkl"
    save_vectorizer(tfidf, vectorizer_path)
    mlflow.log_artifact(vectorizer_path)

    best_model_name = None
    best_model = None
    best_score = 0  # Based on f1-score

    print("\nModel Evaluation Results:")
    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            print(f"\n--- {name.upper()} ---")
            metrics = evaluate_model(model, X_test, y_test)

            # Log model name and metrics
            mlflow.log_param("model", name)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Save and log model
            model_path = f"{model_dir}/{name.lower()}.pkl"
            save_models({name: model}, model_dir)
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(model_path)

            # Select best model (based on f1-score)
            if metrics["f1-score"] > best_score:
                best_score = metrics["f1-score"]
                best_model_name = name
                best_model = model

    # Log best model details in parent run
    mlflow.log_param("best_model", best_model_name)
    print(f"\n✅ Best model selected: {best_model_name.upper()} with F1-score = {best_score:.4f}")

    # -------------------------------
    # Register the best model
    # -------------------------------
    client = MlflowClient()
    signature = infer_signature(X_test, best_model.predict(X_test))
    registered_model_name = "FakeNewsBestModel"

    # Log and register
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        registered_model_name=registered_model_name,
        signature=signature
    )

    print(f"📌 Registered best model: {registered_model_name}")

    #Transition to STAGING
    latest_version = client.get_latest_versions(name=registered_model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=registered_model_name,
        version=latest_version,
        stage="Staging"
    )

    print(f"Transitioned model to 'Staging' in MLflow Model Registry.")
    print("All models logged and best model registered with MLflow.")
