import os
import logging
import pandas as pd
import numpy as np
from typing import Annotated, Tuple

# ML Frameworks
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# ZenML
from zenml import pipeline, step

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des accès Cloud / MinIO
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["AWS_ENDPOINT_URL"] = "http://127.0.0.1:9000"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_VERIFY_SSL"] = "0"

@step
def ingest_agri_data() -> pd.DataFrame:
    """Ingestion des données depuis le bucket MinIO."""
    path = "s3://agri-resillience-bucket/final_scoring.parquet"
    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL")},
        "config_kwargs": {"s3": {"addressing_style": "path"}}
    }
    
    try:
        df = pd.read_parquet(path, storage_options=storage_options)
        logger.info("✅ Données ingérées avec succès depuis MinIO.")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'ingestion : {e}")
        raise e

@step
def preprocess_scoring_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Prétraitement, Équilibrage et Split des données."""
    features = ['final_precipitation', 'ph_level', 'nitrogen_content', 'organic_matter']
    target = 'loan_status'

    # Nettoyage
    for col in features:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=[target])

    X = df[features]
    y = df[target]

    # Encoding de la cible (Nécessaire pour le classifier)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split AVANT oversampling pour éviter le data leakage
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Équilibrage sur le set d'entraînement uniquement
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_raw, y_train_raw)

    # Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_resampled), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=features)
    
    y_train = pd.Series(y_resampled)
    y_test = pd.Series(y_test_raw)

    logger.info("✅ Prétraitement terminé : Train set équilibré.")
    return X_train, X_test, y_train, y_test

@step(experiment_tracker="mlflow_tracker")
def train_voting_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> VotingClassifier:
    """Entraînement du VotingClassifier (Modèle de ton Master)."""
    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced')),
            ('dt', DecisionTreeClassifier())
        ],
        voting='soft',
        weights=[1, 3, 1] 
    )
    
    model.fit(X_train, y_train)
    logger.info("🚀 VotingClassifier entraîné et prêt !")
    return model

@step
def evaluate_resilience_model(
    model: VotingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "f1_score"],
]:
    """Évaluation finale des performances."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    print("\n" + "="*40)
    print(f"📊 PERFORMANCE DU MODÈLE DE RÉSILIENCE")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("="*40 + "\n")
    
    return accuracy, f1

@pipeline(name="mon_projet_resilience_pipeline")
def agri_resilience_pipeline():
    """Architecture MLOps complète."""
    # 1. Ingestion
    df = ingest_agri_data()
    
    # 2. Prétraitement (Synchronisé avec 4 sorties)
    X_train, X_test, y_train, y_test = preprocess_scoring_data(df)
    
    # 3. Entraînement
    model = train_voting_model(X_train, y_train)
    
    # 4. Évaluation
    evaluate_resilience_model(model, X_test, y_test)

if __name__ == "__main__":
    # Lancement du pipeline
    agri_resilience_pipeline()