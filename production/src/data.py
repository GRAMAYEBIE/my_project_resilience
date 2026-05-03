import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import storage_cfg, model_cfg

# On utilise le logger configuré dans logging_utils
logger = logging.getLogger(__name__)

class DataManager:
    """
    Gère le cycle de vie des données : Chargement S3, Nettoyage et Split.
    Adapté pour la production avec gestion d'erreurs et stratification.
    """

    def __init__(self):
        self.s3_path = storage_cfg.S3_PATH
        self.target = model_cfg.target
        self.features = model_cfg.features

    def load_from_s3(self) -> pd.DataFrame:
        try:
            logger.info("📡 Connexion au Minio local : %s", storage_cfg.S3_ENDPOINT)
            
            # Configuration robuste pour S3FS / Pandas
            storage_options = {
                "key": storage_cfg.S3_ACCESS_KEY,
                "secret": storage_cfg.S3_SECRET_KEY,
                "client_kwargs": {
                    "endpoint_url": "http://127.0.0.1:9000",
                    "region_name": "us-east-1",
                },
                "config_kwargs": {
                    "s3": {"addressing_style": "path"} # Important pour MinIO
                }
            }
            
            # Lecture du fichier Parquet sur MinIO
            df = pd.read_parquet(self.s3_path, storage_options=storage_options)
            
            logger.info("✅ Données chargées avec succès : %s lignes", len(df))
            return df
        except Exception as e:
            logger.error("❌ Impossible de joindre Minio : %s", e)
            raise

    def preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les corrections du notebook (Imputation)."""
        df_clean = df.copy()
        
        # 1. Imputation de la précipitation par la médiane
        if 'final_precipitation' in df_clean.columns:
            median_val = df_clean['final_precipitation'].median()
            df_clean['final_precipitation'] = df_clean['final_precipitation'].fillna(median_val)
            logger.info("🩹 Imputation : Médiane appliquée sur 'final_precipitation' (%s)", median_val)

        # 2. Suppression des lignes avec NaN restants sur les colonnes clés
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=self.features + [self.target])
        dropped_count = initial_count - len(df_clean)
        
        if dropped_count > 0:
            logger.warning("🧹 Nettoyage : %s lignes supprimées à cause de valeurs manquantes.", dropped_count)
            
        return df_clean

    def get_train_test_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Sépare les features/target et effectue le split stratifié."""
        X = df[self.features]
        y = df[self.target]

        # Stratify=y est CRUCIAL pour ne pas perdre la classe 'PREMIUM_ELIGIBLE'
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=model_cfg.test_size, 
            random_state=model_cfg.random_state,
            stratify=y 
        )

        logger.info("⚖️ Split terminé (Stratifié) | Train: %s - Test: %s", X_train.shape[0], X_test.shape[0])
        return X_train, X_test, y_train, y_test