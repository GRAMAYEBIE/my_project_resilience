import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_STORAGE = Path(os.environ.get("DATA_STORAGE", "./data_storage"))
MODEL_STORAGE = Path(os.environ.get("MODEL_STORAGE", "./model_storage"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    processed = DATA_STORAGE / "processed"
    features_path = processed / "train_features.csv"
    labels_path = processed / "train_labels.csv"
    pre_blueprint_path = MODEL_STORAGE / "preprocessor_blueprint.joblib"

    # 1. Vérifications de sécurité
    if not all([features_path.exists(), labels_path.exists(), pre_blueprint_path.exists()]):
        logger.error("❌ Artifacts manquants. Lancez d'abord le 'feature_pipeline'.")
        return

    MODEL_STORAGE.mkdir(parents=True, exist_ok=True)

    # 2. Chargement des données
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]  # On récupère la première colonne comme target

    # 3. Split stratifié (Crucial pour tes classes déséquilibrées)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Prétraitement (Fit sur Train uniquement)
    logger.info("🧪 Application du prétraitement...")
    preprocessor = joblib.load(pre_blueprint_path)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 5. Gestion du déséquilibre (SMOTE)
    # Puisque tu as des classes très rares (Premium), SMOTE va aider le modèle
    logger.info("⚖️ Application du SMOTE pour équilibrer les classes...")
    smote = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # 6. Entraînement du Modèle Champion (VotingClassifier)
    logger.info("🏋️ Entraînement du VotingClassifier (RF + XGB)...")
    
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    model = VotingClassifier(
        estimators=[('rf', clf1), ('xgb', clf2)],
        voting='soft'  # 'soft' permet de récupérer les probabilités pour l'Inference API
    )
    
    model.fit(X_train_res, y_train_res)

    # 7. Évaluation
    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    logger.info(f"⭐ Précision globale : {acc:.4f}")

    # 8. Sauvegarde des Artifacts (Noms synchronisés avec l'Inference API)
    logger.info("💾 Sauvegarde des artifacts finaux...")
    
    joblib.dump(model, MODEL_STORAGE / "voting_model_champion.joblib")
    joblib.dump(preprocessor, MODEL_STORAGE / "scaler.joblib") # On sauvegarde le fitted scaler
    
    # Métriques au format JSON
    metrics = {
        "accuracy": acc,
        "model_type": "VotingClassifier(RF, XGB)",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_samples_train": len(X_train_res),
        "target_column": "loan_status"
    }
    
    (MODEL_STORAGE / "metrics.json").write_text(json.dumps(metrics, indent=4))
    (MODEL_STORAGE / "classification_report.txt").write_text(report)

    logger.info("✅ Training pipeline terminé avec succès.")

if __name__ == "__main__":
    main()