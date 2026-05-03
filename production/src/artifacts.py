import json
import logging
from pathlib import Path
from typing import Union, Any

import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import des chemins depuis ton config
from src.config import paths

logger = logging.getLogger(__name__)

# Type personnalisé pour accepter du texte ou des objets Path
PathLike = Union[str, Path]

def _to_path(path: PathLike) -> Path:
    """Convertit une entrée en objet Path Windows/Linux."""
    return path if isinstance(path, Path) else Path(path)

def _ensure_parent_dir(path: Path) -> None:
    """Crée le dossier parent s'il n'existe pas (évite les crashs)."""
    path.parent.mkdir(parents=True, exist_ok=True)

# --- FONCTIONS DE SAUVEGARDE (SAVE) ---

def save_model(model: Any, path: PathLike = paths.model_path) -> None:
    """Sauvegarde le VotingClassifier Champion."""
    path = _to_path(path)
    _ensure_parent_dir(path)
    logger.info("💾 Sauvegarde du modèle champion : %s", path)
    joblib.dump(model, path)

def save_scaler(scaler: StandardScaler, path: PathLike = paths.scaler_path) -> None:
    """Sauvegarde le StandardScaler entraîné."""
    path = _to_path(path)
    _ensure_parent_dir(path)
    logger.info("📏 Sauvegarde du scaler : %s", path)
    joblib.dump(scaler, path)

def save_encoder(encoder: LabelEncoder, path: PathLike = paths.encoder_path) -> None:
    """Sauvegarde le LabelEncoder (crucial pour décoder HIGH_RISK, etc.)."""
    path = _to_path(path)
    _ensure_parent_dir(path)
    logger.info("🏷️ Sauvegarde du LabelEncoder : %s", path)
    joblib.dump(encoder, path)

def save_metrics(metrics: dict, path: PathLike = paths.metrics_path) -> None:
    """Sauvegarde les scores (F1, Accuracy) au format JSON."""
    path = _to_path(path)
    _ensure_parent_dir(path)
    logger.info("📊 Sauvegarde des métriques : %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

# --- FONCTIONS DE CHARGEMENT (LOAD) ---

def load_model(path: PathLike = paths.model_path) -> Any:
    """Charge le modèle pour l'inférence ou Streamlit."""
    path = _to_path(path)
    logger.info("📂 Chargement du modèle : %s", path)
    return joblib.load(path)

def load_scaler(path: PathLike = paths.scaler_path) -> StandardScaler:
    path = _to_path(path)
    logger.info("📂 Chargement du scaler : %s", path)
    return joblib.load(path)

def load_encoder(path: PathLike = paths.encoder_path) -> LabelEncoder:
    path = _to_path(path)
    logger.info("📂 Chargement du LabelEncoder : %s", path)
    return joblib.load(path)