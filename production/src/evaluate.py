import logging
import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.config import paths

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Responsable de l'évaluation du modèle et de la persistance des scores.
    """

    def evaluate(self, y_true, y_pred) -> dict:
        """
        Calcule les métriques et génère le rapport complet.
        """
        logger.info("📊 Calcul des performances du modèle...")
        
        # 1. Calcul des scores principaux
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # 2. Rapport détaillé (celui que tu voyais dans ton terminal)
        report = classification_report(y_true, y_pred)
        
        logger.info("⭐ Overall Accuracy: %.4f", acc)
        logger.info("⭐ Weighted F1-Score: %.4f", f1_weighted)
        print("\n" + "="*40)
        print("RAPPORT DE CLASSIFICATION")
        print("="*40)
        print(report)
        print("="*40)

        # 3. Préparation des métriques pour sauvegarde JSON
        metrics = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_weighted, 4)
        }
        
        return metrics

    def save_results(self, metrics: dict):
        """
        Sauvegarde les scores dans un fichier JSON pour le monitoring.
        """
        with open(paths.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("📊 Métriques sauvegardées dans : %s", paths.metrics_path)