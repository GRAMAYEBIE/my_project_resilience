from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Dict, Any

@dataclass(frozen=True)
class PathConfig:
    """Gère l'arborescence du projet."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    artifacts_dir: Path = base_dir / "artifacts"
    logs_dir: Path = base_dir / "logs"
    
    # Fichiers
    model_path: Path = artifacts_dir / "voting_model_champion.joblib"
    scaler_path: Path = artifacts_dir / "scaler.joblib"
    encoder_path: Path = artifacts_dir / "label_encoder.joblib"
    metrics_path: Path = artifacts_dir / "metrics.json"

    def __post_init__(self):
        # Création des dossiers si inexistants
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class ModelConfig:
    """Configuration du VotingClassifier et du SMOTE."""
    random_state: int = 42
    test_size: float = 0.2
    k_neighbors_smote: int = 1
    
    features: List[str] = field(default_factory=lambda: [
        'final_precipitation', 'ph_level', 'nitrogen_content', 'organic_matter'
    ])
    target: str = "loan_status"
    
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200, 'max_depth': 12, 'class_weight': 'balanced', 'random_state': 42
    })
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'eval_metric': 'mlogloss', 'random_state': 42
    })
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_iter': 2000, 'class_weight': 'balanced', 'random_state': 42
    })

@dataclass(frozen=True)
class StorageConfig:
    # On récupère l'IP ou le nom, et on nettoie tout caractère parasite
    _endpoint = os.getenv("S3_ENDPOINT_URL", "http://172.19.0.3:9000")
    S3_ENDPOINT: str = str(_endpoint).strip().strip('"').strip("'").rstrip('/')
    
    S3_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY_ID", "admin").strip()
    S3_SECRET_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "password123").strip()
    
    S3_BUCKET: str = "agri-resillience-bucket"
    S3_FILE: str = "final_scoring.parquet"
    S3_PATH: str = f"s3://{S3_BUCKET}/{S3_FILE}"
# Initialisation des instances
paths = PathConfig()
model_cfg = ModelConfig()
storage_cfg = StorageConfig()