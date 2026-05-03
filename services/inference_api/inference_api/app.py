import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- CONFIGURATION ---
MODEL_STORAGE = Path(os.environ.get("MODEL_STORAGE", "/app/model_storage"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agriculture Resilience Inference API",
    version="2.0",
    description="API de prédiction du statut d'éligibilité aux prêts agricoles.",
)

# --- MODELS PYDANTIC ---
class FeaturesRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence_score: float
    all_probabilities: Dict[str, float]

# --- CHARGEMENT DES ARTIFACTS ---
_model = None
_scaler = None
_label_encoder = None
_feature_names: List[str] = []

def _load_artifacts() -> None:
    global _model, _scaler, _label_encoder, _feature_names
    
    # On force les chemins absolus pour Docker
    path_model = MODEL_STORAGE / "model.joblib"
    path_scaler = MODEL_STORAGE / "scaler.joblib"
    path_encoder = MODEL_STORAGE / "label_encoder.joblib"

    try:
        # On charge les 3 piliers séparément
        if path_model.exists() and path_scaler.exists() and path_encoder.exists():
            _model = joblib.load(path_model)
            _scaler = joblib.load(path_scaler)
            _label_encoder = joblib.load(path_encoder)
            
            # Récupération des colonnes
            if hasattr(_model, "feature_names_in_"):
                _feature_names = list(_model.feature_names_in_)
            else:
                # Si le modèle ne les a pas, on les définit manuellement
                _feature_names = ["rainfall", "ph_level", "nitrogen_content", "organic_matter"]
                
            logger.info("✅ Pipeline chargé : Modèle + Scaler + Encoder (V1.3.2)")
        else:
            logger.error("⚠️ Un des fichiers artifacts est manquant dans /app/model_storage")
            
    except Exception as e:
        logger.error(f"❌ Erreur critique : {e}")

@app.on_event("startup")
def startup() -> None:
    _load_artifacts()

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    status_icon = "✅" if _model else "❌"
    return f"""
    <html><body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
        <h1>🌾 Agri-Resilience API</h1>
        <p>Statut du modèle : {status_icon}</p>
        <p>Scaler : {"✅" if _scaler else "❌"} | Encoder : {"✅" if _label_encoder else "❌"}</p>
        <a href="/docs">Accéder à la Documentation Swagger</a>
    </body></html>
    """

@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if _model else "initializing",
        "has_scaler": _scaler is not None,
        "has_label_encoder": _label_encoder is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: FeaturesRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        data = req.features
        # LOG DE DÉBOGAGE : Regarde tes logs Docker pour voir ça
        logger.info(f"📥 Données reçues du Frontend : {data}")
        
        # 1. ALIGNEMENT STRICT (On mappe les noms de ton UI vers ton modèle)
        # Ton UI envoie 'final_precipitation', 'ph_level', etc.
        try:
            vals = [
                float(data['final_precipitation']),
                float(data['ph_level']),
                float(data['nitrogen_content']),
                float(data['organic_matter'])
            ]
        except KeyError as e:
            logger.error(f"❌ Clé manquante dans la requête : {e}")
            # Repli sécurisé si les clés ne correspondent pas exactement
            vals = [
                float(data.get("final_precipitation", 0)),
                float(data.get("ph_level", 0)),
                float(data.get("nitrogen_content", 0)),
                float(data.get("organic_matter", 0))
            ]

        X_input = np.array([vals])
        logger.info(f"🔢 Array avant Scaling : {X_input}")

        # 2. SCALING (OBLIGATOIRE)
        # Si tu ne scales pas, pH 4.0 est interprété comme une valeur énorme
        X_final = _scaler.transform(X_input) if _scaler else X_input
        logger.info(f"📏 Array après Scaling : {X_final}")

        # 3. PRÉDICTION
        prediction_idx = int(_model.predict(X_final)[0])
        
        # 4. CONFIANCE (Probabilités)
        confidence = 0.5
        all_probs = {}
        if hasattr(_model, "predict_proba"):
            probas = _model.predict_proba(X_final)[0]
            confidence = float(np.max(probas))
            if _label_encoder:
                labels = _label_encoder.classes_
                all_probs = {str(labels[i]): float(probas[i]) for i in range(len(probas))}

        # 5. DÉCODAGE DU LABEL
        predicted_label = str(_label_encoder.inverse_transform([prediction_idx])[0]) if _label_encoder else "Indéfini"

        logger.info(f"✅ Résultat : {predicted_label} ({confidence*100:.2f}%)")

        return PredictionResponse(
            predicted_class=prediction_idx,
            predicted_label=predicted_label,
            confidence_score=confidence,
            all_probabilities=all_probs
        )

    except Exception as e:
        logger.error(f"💥 Erreur Critique : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")