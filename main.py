from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from schemas import PredictRequest, PredictResponse

# 1. Paths and Global Variables
MODEL_PATH = "model_files/svc_trained_model.pkl"
PCLASS_ENC_PATH = "model_files/pclass_encoder.pkl"
GENDER_ENC_PATH = "model_files/gender_encoder.pkl"
SIBLING_ENC_PATH = "model_files/sibling_encoder.pkl"
EMBARKED_ENC_PATH = "model_files/embarked_encoder.pkl"

model = pclass_le = gender_le = sibling_le = embarked_le = None

# 2. Define the loading function
def load_artifacts():
    global model, pclass_le, gender_le, sibling_le, embarked_le
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
    with open(PCLASS_ENC_PATH, 'rb') as f: pclass_le = pickle.load(f)
    with open(GENDER_ENC_PATH, 'rb') as f: gender_le = pickle.load(f)
    with open(SIBLING_ENC_PATH, 'rb') as f: sibling_le = pickle.load(f)
    with open(EMBARKED_ENC_PATH, 'rb') as f: embarked_le = pickle.load(f)

# 3. Lifespan handles the startup call
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
        print("Successfully loaded all model artifacts!")
    except Exception as e:
        print(f"Loading failed: {e}")
    yield

app = FastAPI(title="Titanic Predictor", lifespan=lifespan)

# 4. Your endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([{
            'PClass': req.PClass,
            'Gender': req.Gender,
            'Sibling': req.Sibling,
            'Embarked': req.Embarked
        }])

        df['PClass'] = pclass_le.transform(df['PClass'])
        df['Gender'] = gender_le.transform(df['Gender'])
        df['Sibling'] = sibling_le.transform(df['Sibling'])
        df['Embarked'] = embarked_le.transform(df['Embarked'])

        features = df[['PClass', 'Gender', 'Sibling', 'Embarked']].to_numpy()
        pred = model.predict(features)
        pred_value = int(np.ravel(pred)[0])

        label = "SURVIVED" if pred_value == 1 else "NOT SURVIVED"
        return PredictResponse(predicted_label=pred_value, prediction=label)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")