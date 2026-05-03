import os
import pandas as pd
import numpy as np
import io
from minio import Minio
from zenml import step, pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
MINIO_URL = "127.0.0.1:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "password123"
BUCKET_NAME = "agriculture-bronze"
DATA_DIR = "data/zindi_raw"

class MinioHandler:
    def __init__(self):
        self.client = Minio(MINIO_URL, ACCESS_KEY, SECRET_KEY, secure=False)
        if not self.client.bucket_exists(BUCKET_NAME):
            self.client.make_bucket(BUCKET_NAME)

    def upload_parquet(self, df, object_name):
        """Converts DataFrame to Parquet and uploads to MinIO."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
        buffer.seek(0)
        self.client.put_object(
            BUCKET_NAME, object_name, buffer, len(buffer.getvalue()),
            content_type="application/octet-stream"
        )

    def upload_raw(self, local_path, object_name):
        """Uploads a file without conversion (e.g., .txt)."""
        self.client.fput_object(BUCKET_NAME, object_name, local_path)

@step
def bronze_ingestion_step() -> str:
    """ZenML Step to ingest all raw data into MinIO Bronze Layer."""
    handler = MinioHandler()
    
    # 1. PROCESS TABULAR DATA & METADATA (CSV and TXT)
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        
        if os.path.isfile(item_path):
            if item.endswith(".csv"):
                df = pd.read_csv(item_path)
                # Store in tabular folder
                target_path = f"tabular/{item.replace('.csv', '.parquet')}"
                handler.upload_parquet(df, target_path)
                print(f"✅ CSV Converted & Uploaded: {target_path}")
            
            elif item.endswith(".txt"):
                # Store in metadata folder
                target_path = f"metadata/{item}"
                handler.upload_raw(item_path, target_path)
                print(f"✅ TXT Uploaded: {target_path}")

    # 2. PROCESS IMAGE FOLDERS (TRAIN & TEST)
    image_folders = ["image_arrays_train"]
    
    for folder in image_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.exists(folder_path):
            print(f"📦 Processing folder: {folder}")
            all_images = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            batch_size = 250
            
            for i in range(0, len(all_images), batch_size):
                batch_files = all_images[i:i + batch_size]
                batch_data = []
                
                for img_name in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}"):
                    arr = np.load(os.path.join(folder_path, img_name))
                    batch_data.append({
                        "image_id": img_name.split('.')[0],
                        "data": arr.tolist()
                    })
                
                df_batch = pd.DataFrame(batch_data)
                # Keep folder structure in MinIO
                object_name = f"images/{folder}/batch_{i//batch_size + 1}.parquet"
                handler.upload_parquet(df_batch, object_name)
    
    return "Bronze Layer successfully synced with ZenML orchestration."

@pipeline
def agriculture_resilience_bronze_pipeline():
    """Main ZenML Pipeline for the Bronze Layer."""
    bronze_ingestion_step()

if __name__ == "__main__":
    # Ensure ZenML is initialized before running: 'zenml init'
    agriculture_resilience_bronze_pipeline()