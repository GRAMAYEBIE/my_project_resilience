import pandas as pd
import s3fs
from sqlalchemy import create_engine
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Configuration (Matching your MinIO environment)
MINIO_URL = "http://localhost:9000"
BUCKET_NAME = "agri-resillience-bucket"  # Validated with double 'll'
FILE_NAME = "final_scoring.parquet"
GOLD_PATH = f"{BUCKET_NAME}/{FILE_NAME}"

# 2. PostgreSQL Configuration (Warehouse)
# Format: postgresql://[user]:[password]@[host]:[port]/[database]
DB_URL = "postgresql://user:password@localhost:5432/agriculture_db"

def start_transfer():
    """Extracts Gold data from MinIO and loads it into the PostgreSQL Warehouse."""
    
    # Initialize S3FS for MinIO connection
    fs = s3fs.S3FileSystem(
        key="admin",
        secret="password123",
        client_kwargs={"endpoint_url": MINIO_URL}
    )

    try:
        logger.info(f"🔍 Searching for Gold file: {GOLD_PATH}...")
        
        # Check if file exists before reading
        if not fs.exists(GOLD_PATH):
            logger.error("❌ Gold file not found. Please check the bucket name in MinIO.")
            return

        # Load Parquet file into memory
        with fs.open(GOLD_PATH, 'rb') as f:
            df = pd.read_parquet(f)
        
        logger.info(f"📊 Data loaded successfully ({len(df)} rows).")

        # 3. Injection into PostgreSQL
        logger.info("🚀 Sending data to PostgreSQL (Table: fact_scoring_results)...")
        engine = create_engine(DB_URL)
        
        # 'replace' ensures the table is updated with the latest scores from your pipeline
        df.to_sql("fact_scoring_results", engine, if_exists="replace", index=False)
        
        logger.info("✅ SUCCESS! Your Data Warehouse is now up to date.")

    except Exception as e:
        logger.error(f"💥 Transfer failed: {e}")

if __name__ == "__main__":
    start_transfer()