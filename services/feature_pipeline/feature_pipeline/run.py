import sys
import os
import json
import logging
from pathlib import Path
import joblib
import pandas as pd
from zenml import pipeline, step
from sklearn.pipeline import Pipeline

# --- GESTION DES CHEMINS (ROOT) ---
current_file = Path(__file__).resolve()
# On remonte jusqu'à 'mon_projet_resilience'
project_root = current_file.parents[4] if "production" in str(current_file) else current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from production.src.data import DataManager

# Configuration des volumes partagés (Docker)
DATA_STORAGE = Path(os.environ.get("DATA_STORAGE", "./data_storage"))
MODEL_STORAGE = Path(os.environ.get("MODEL_STORAGE", "./model_storage"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@step
def sync_production_artifacts() -> None:
    """ 
    Récupère les artifacts déjà générés par la phase d'entraînement 'production' 
    pour les rendre disponibles aux services Docker (Inference, UI).
    """
    prod_artifacts = project_root / "production" / "artifacts"
    MODEL_STORAGE.mkdir(parents=True, exist_ok=True)
    
    files_to_sync = [
        "voting_model_champion.joblib",
        "scaler.joblib",
        "label_encoder.joblib",
        "metrics.json"
    ]
    
    for file in files_to_sync:
        src = prod_artifacts / file
        if src.exists():
            dest = MODEL_STORAGE / file
            # On copie le fichier pour que le volume Docker le possède
            import shutil
            shutil.copy2(src, dest)
            logger.info(f"🔄 Artifact synchronisé : {file}")
        else:
            logger.warning(f"⚠️ Artifact source introuvable : {file}")

@step
def extract_and_store_data() -> str:
    """Extraction depuis MinIO via ton DataManager sécurisé."""
    raw_dir = DATA_STORAGE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "agriculture_resilience_raw.csv"

    # Ton DataManager utilise déjà 127.0.0.1 ou l'IP Docker selon ta config
    data_mgr = DataManager()
    try:
        df = data_mgr.load_from_s3()
        df.to_csv(raw_path, index=False)
        logger.info(f"✅ Données extraites de MinIO ({len(df)} lignes) -> {raw_path}")
        return str(raw_path)
    except Exception as e:
        logger.error(f"💥 Erreur lors de l'extraction : {e}")
        raise e

@step
def process_and_create_schema(raw_path_str: str) -> None:
    """Prépare le terrain pour l'API d'Inférence."""
    raw_path = Path(raw_path_str)
    processed_dir = DATA_STORAGE / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    
    # On utilise le nom exact de ta target après transformation Gold
    target_col = "loan_status" 
    
    X = df.drop(columns=[target_col]) if target_col in df.columns else df
    
    # Génération du Schema JSON (le contrat technique de ton API)
    schema = {
        "feature_names": X.columns.tolist(),
        "target": target_col,
        "n_features": len(X.columns),
        "categorical_features": X.select_dtypes(include=['object']).columns.tolist(),
        "numerical_features": X.select_dtypes(exclude=['object']).columns.tolist()
    }
    
    schema_path = MODEL_STORAGE / "feature_schema.json"
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=4)
    
    logger.info(f"✅ Schéma technique généré : {len(schema['feature_names'])} features détectées.")

@pipeline
def agriculture_resilience_feature_pipeline():
    """Pipeline ZenML maître pour la gestion des features."""
    sync_production_artifacts() # Étape clé : lie ton travail local au monde Docker
    raw_path = extract_and_store_data()
    process_and_create_schema(raw_path)

if __name__ == "__main__":
    agriculture_resilience_feature_pipeline()