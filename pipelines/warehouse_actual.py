import pandas as pd
import s3fs
from sqlalchemy import create_engine

# 1. Paths & Connections
S3_ENDPOINT = "http://localhost:9000"
# Double 'll' as confirmed in your MinIO
GOLD_PATH = "s3://agri-resillience-bucket/final_scoring.parquet"
# Your Warehouse Credentials
DB_URL = "postgresql://user:password@localhost:5432/agriculture_db"

def start_transfer():
    # Initialize MinIO Access
    fs = s3fs.S3FileSystem(
        key="admin",
        secret="password123",
        client_kwargs={'endpoint_url': S3_ENDPOINT}
    )

    print("🔍 Fetching Gold Data...")
    with fs.open(GOLD_PATH, mode='rb') as f:
        df = pd.read_parquet(f)

    # 2. SQL Injection
    print(f"🚀 Loading {len(df)} rows into PostgreSQL Warehouse...")
    engine = create_engine(DB_URL)
    df.to_sql("final_scoring_results", engine, if_exists="replace", index=False)
    
    print("✅ Warehouse Updated!")

if __name__ == "__main__":
    start_transfer()