#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

# Imports de tes nouveaux modules
from src.config import paths, model_cfg
from src.logging_utils import setup_logging
from src.data import DataManager
from src.preprocess import Preprocessor
from src.train import ModelTrainer
from src.evaluate import Evaluator
from src.artifacts import (
    save_model, save_scaler, save_encoder, 
    save_metrics, load_model, load_scaler, load_encoder
)

logger = logging.getLogger(__name__)

def train_pipeline(args: argparse.Namespace) -> None:
    """Orchestre le chargement, le prétraitement, l'entraînement et l'évaluation."""
    logger.info("🎬 Démarrage du Pipeline d'Entraînement...")

    # 1. Chargement et Split (via MinIO/S3)
    data_mgr = DataManager()
    df_raw = data_mgr.load_from_s3()
    df_clean = data_mgr.preprocess_raw_data(df_raw)
    X_train, X_test, y_train, y_test = data_mgr.get_train_test_data(df_clean)

    # 2. Prétraitement (Fit sur Train, Transform sur les deux)
    pre = Preprocessor()
    pre.fit(X_train, y_train)
    X_train_scaled, y_train_encoded = pre.transform(X_train, y_train)
    X_test_scaled, y_test_encoded = pre.transform(X_test, y_test)

    # 3. Entraînement (SMOTE + VotingClassifier)
    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train_encoded)

    # 4. Évaluation
    y_pred_encoded = trainer.model.predict(X_test_scaled)
    evaluator = Evaluator()
    metrics = evaluator.evaluate(y_test_encoded, y_pred_encoded)

    # 5. Sauvegarde des Artifacts (via artifacts.py)
    save_model(trainer.model)
    save_scaler(pre.scaler)
    save_encoder(pre.label_encoder)
    save_metrics(metrics)
    
    logger.info("✅ Pipeline terminé. Tous les artifacts sont dans : %s", paths.artifacts_dir)


def predict_pipeline(args: argparse.Namespace) -> None:
    """Charge les artifacts et simule une prédiction sur un échantillon."""
    logger.info("🔮 Démarrage du Pipeline de Prédiction...")

    # Chargement des outils
    model = load_model()
    scaler = load_scaler()
    encoder = load_encoder()

    # Simulation : On recharge les données pour prendre un exemple (juste pour le test)
    data_mgr = DataManager()
    df = data_mgr.load_from_s3()
    sample_df = df[model_cfg.features].iloc[[0]]
    
    # Prédire
    sample_scaled = scaler.transform(sample_df)
    pred_code = model.predict(sample_scaled)
    pred_label = encoder.inverse_transform(pred_code)[0]

    print("\n" + "=" * 50)
    print("🚀 TEST DE PRÉDICTION UNITAIRE")
    print("=" * 50)
    print(f"Entrée : {sample_df.to_dict(orient='records')[0]}")
    print(f"Résultat prédit : {pred_label}")
    print("=" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de Résilience Agricole - Production Ready")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log (DEBUG, INFO, ...)")
    
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Commande Train
    p_train = sub.add_parser("train", help="Lance l'entraînement complet.")
    p_train.set_defaults(func=train_pipeline)
    
    # Commande Predict
    p_pred = sub.add_parser("predict", help="Teste le modèle sauvegardé.")
    p_pred.set_defaults(func=predict_pipeline)
    
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        args.func(args)
    except Exception as e:
        logger.error("💥 Le pipeline a échoué : %s", e, exc_info=True)

if __name__ == "__main__":
    main()