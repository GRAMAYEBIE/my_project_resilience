import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import paths

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self):
        """
        L'initialisation prépare les compartiments de notre boîte à outils.
        Même s'ils sont vides au début, ils appartiennent à l'objet (self).
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        logger.info("🛠️ Preprocessor initialisé (Scaler & LabelEncoder prêts).")

    def fit(self, X_train, y_train):
        """
        Apprend les statistiques des données (Moyenne, Écart-type, Classes).
        """
        logger.info("🧪 Apprentissage des paramètres de transformation (Fit)...")
        
        # Le scaler apprend les min/max des features
        self.scaler.fit(X_train)
        
        # L'encodeur apprend les noms des classes (ex: 'HIGH_RISK' -> 0)
        self.label_encoder.fit(y_train)
        
        logger.info("✅ Apprentissage terminé. Classes : %s", list(self.label_encoder.classes_))

    def transform(self, X, y=None):
        """
        Applique la transformation apprise lors du fit.
        """
        X_scaled = self.scaler.transform(X)
        
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
            
        return X_scaled

    def save_artifacts(self):
        """
        Sauvegarde les outils entraînés pour que l'App Streamlit puisse les réutiliser.
        """
        # On utilise les chemins définis dans ton config.py
        joblib.dump(self.scaler, paths.scaler_path)
        joblib.dump(self.label_encoder, paths.encoder_path)
        
        logger.info("💾 Artifacts de prétraitement sauvegardés dans : %s", paths.artifacts_dir)