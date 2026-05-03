import logging
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.config import paths, model_cfg

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """Initialise le Champion (VotingClassifier) avec les réglages du config."""
        logger.info("🏗️ Initialisation du ModelTrainer (Architecture: VotingClassifier)")
        
        # 1. SMOTE pour gérer le déséquilibre (Premium class)
        self.smote = SMOTE(
            k_neighbors=model_cfg.k_neighbors_smote, 
            random_state=model_cfg.random_state
        )
        
        # 2. Les 3 Mousquetaires
        self.rf = RandomForestClassifier(**model_cfg.rf_params)
        self.xgb = XGBClassifier(**model_cfg.xgb_params)
        self.lr = LogisticRegression(**model_cfg.lr_params)
        
        # 3. Le Champion Final
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('lr', self.lr)
            ],
            voting='soft'
        )

    def train(self, X_train, y_train):
        """Applique SMOTE et entraîne le VotingClassifier."""
        logger.info("⚖️ Application du SMOTE pour équilibrer les classes...")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        
        logger.info("🏋️ Entraînement du VotingClassifier en cours...")
        self.model.fit(X_res, y_res)
        logger.info("✅ Entraînement terminé.")

    def save_model(self):
        """Sauvegarde le modèle entraîné."""
        joblib.dump(self.model, paths.model_path)
        logger.info("🏆 Modèle champion sauvegardé dans : %s", paths.model_path)