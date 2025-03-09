from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "App Rating Prediction API"}

@app.post("/predict/")
def predict(reviews: int, rating: float):
    input_data = np.array([[reviews, rating]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "High Rating" if prediction == 1 else "Low Rating"
    return {"prediction": result}
