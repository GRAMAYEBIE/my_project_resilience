import pandas as pd
import boto3
import io
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from botocore.client import Config
from src.config import train_params, warehouse_cfg

class DataLoader:
    def __init__(self):
        self.test_size = train_params.test_size
        self.random_state = train_params.random_state
        # We initialize a client to avoid OSError with pandas read_parquet on Windows
        self.s3_client = boto3.client(
            's3',
            endpoint_url='http://127.0.0.1:9000',
            aws_access_key_id='admin',
            aws_secret_access_key='password123',
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
            region_name='us-east-1'
        )

    # --- ADDED: The missing method main.py is calling ---
    def load_data(self) -> pd.DataFrame:
        """
        Orchestrator method required by main.py.
        Tries S3 first, then falls back to Postgres.
        """
        try:
            return self.load_from_s3()
        except Exception as e:
            print(f"⚠️ S3 failed, trying Postgres: {e}")
            return self.load_from_postgres()

    def load_from_s3(self) -> pd.DataFrame:
        """Improved load from MinIO using the S3 client for stability."""
        print(f"📡 Connexion à MinIO: {warehouse_cfg.S3_BUCKET_PATH}")
        
        # Extraction of bucket and key from the path
        # Assuming S3_BUCKET_PATH is s3://agri-resillience-bucket/final_scoring.parquet
        bucket = "agri-resillience-bucket"
        key = "final_scoring.parquet"
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        
        print(f"🔍 columns in Parquet : {df.columns.tolist()}")
        print(f"Nombre de lignes : {len(df)}")
        return df

    def load_from_postgres(self) -> pd.DataFrame:
        """Loads the data from the PostgreSQL database."""
        print("🗄️ Connexion à PostgreSQL...")
        engine = create_engine(warehouse_cfg.DB_URL)
        return pd.read_sql(warehouse_cfg.QUERY, engine)

    def split_data(self, df: pd.DataFrame):
        """Prépare et divise les données avec imputation des valeurs manquantes."""
        print("🩹 Imputation des valeurs manquantes (Médiane pour précipitation)...")
        
        df['final_precipitation'] = df['final_precipitation'].fillna(df['final_precipitation'].median())
        
        if 'credit_score' in df.columns:
            df['credit_score'] = df['credit_score'].fillna(df['credit_score'].mean())

        # Security drop for remaining NaNs in features or target
        clean_df = df.dropna(subset=train_params.features + [train_params.target])

        X = clean_df[train_params.features]
        y = clean_df[train_params.target]
        
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )