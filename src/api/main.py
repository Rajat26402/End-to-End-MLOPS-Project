from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pickle
import os
import logging


logging.basicConfig(level=logging.INFO)

# Set MLflow Tracking URI to point to the running MLflow server
mlflow.set_tracking_uri("http://host.docker.internal:8082")  

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model from MLflow registry
model_name = "FakeNewsBestModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

app = FastAPI()

class NewsItem(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: NewsItem):
    logging.info(f"Received request: {item.text}")
    vectorized = vectorizer.transform([item.text])
    prediction = model.predict(vectorized)
    label = "FAKE" if prediction[0] == 0 else "TRUE"
    logging.info(f"Prediction result: {label}")
    return {"prediction": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8083)
