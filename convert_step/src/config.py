from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class PathConfig:
    """Gère tous les chemins du projet de manière robuste."""
    root_dir: Path = Path(__file__).parent.parent
    data_dir: Path = root_dir / "data"
    out_dir: Path = root_dir / "artifacts"
    
    # Chemins des fichiers
    raw_data_path: Path = data_dir / "raw" / "agri_data.csv"
    model_path: Path = out_dir / "model.joblib"
    scaler_path: Path = out_dir / "scaler.joblib"
    metrics_path: Path = out_dir / "metrics.json"
    report_path: Path = out_dir / "classification_report.txt"

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    features: List[str] = field(default_factory=lambda: [
        'final_precipitation', 
        'ph_level', 
        'nitrogen_content', 
        'organic_matter'
    ])
    target: str = "loan_status"

@dataclass(frozen=True)
class WarehouseConfig:
    S3_BUCKET_PATH: str = "s3://agri-resillience-bucket/final_scoring.parquet"
    DB_URL: str = os.getenv("DB_URL", "postgresql://admin:password123@localhost:5432/agriculture_db")
    QUERY: str = "SELECT * FROM final_scoring_results" 

# --- INITIALISATION DES INSTANCES ---
paths = PathConfig() # <--- C'est lui qui manquait !
train_params = TrainConfig()
warehouse_cfg = WarehouseConfig()