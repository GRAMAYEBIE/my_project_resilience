import pandas as pd
import boto3
import io
import logging
import gc
import numpy as np
from zenml import step, pipeline
from botocore.client import Config
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    return boto3.client(
        's3', 
        endpoint_url='http://127.0.0.1:9000', 
        aws_access_key_id='admin',           
        aws_secret_access_key='password123',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        region_name='us-east-1'
    )

@step(enable_cache=False)
def silver_refinery_engine() -> pd.DataFrame:
    s3 = get_s3_client()
    bronze_bucket = 'agriculture-bronze'
    silver_bucket = 'agriculture-silver'
    
    # --- FIX 1: Ensure Silver bucket exists before anything else ---
    try:
        s3.head_bucket(Bucket=silver_bucket)
    except ClientError:
        logger.info(f"Creating missing bucket: {silver_bucket}")
        s3.create_bucket(Bucket=silver_bucket)

    # --- FIX 2: Removed the 'buffer' line that was causing the crash here ---
    
    # 1. Loading metadata
    logger.info("Loading metadata from Bronze layer...")
    obj_train = s3.get_object(Bucket=bronze_bucket, Key='tabular/Train.parquet')
    df_train = pd.read_parquet(io.BytesIO(obj_train['Body'].read()))
    
    obj_fields = s3.get_object(Bucket=bronze_bucket, Key='tabular/fields_w_additional_info.parquet')
    df_fields = pd.read_parquet(io.BytesIO(obj_fields['Body'].read()))
    
    # Normalize columns to lowercase
    df_train.columns = [c.lower() for c in df_train.columns]
    df_fields.columns = [c.lower() for c in df_fields.columns]
    df_meta = pd.merge(df_train, df_fields, on='field_id', how='inner')

    # 2. Image processing with memory efficiency
    image_prefix = 'images/image_arrays_train/'
    response = s3.list_objects_v2(Bucket=bronze_bucket, Prefix=image_prefix)
    batch_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]
    
    final_shards = []
    
    for key in batch_keys:
        logger.info(f"Optimized reading for batch: {key}")
        obj_img = s3.get_object(Bucket=bronze_bucket, Key=key)
        
        try:
            # Read batch data
            batch_df = pd.read_parquet(io.BytesIO(obj_img['Body'].read()))
            batch_df.columns = [c.lower() for c in batch_df.columns]
            
            if 'image_id' in batch_df.columns:
                batch_df = batch_df.rename(columns={'image_id': 'field_id'})

            # --- MEMORY REDUCTION ---
            # Extract simple statistics to avoid storing heavy multi-dimensional arrays in RAM
            if 'data' in batch_df.columns:
                # Apply mean calculation and drop heavy column immediately
                batch_df['pixel_mean'] = batch_df['data'].apply(lambda x: np.mean(x) if x is not None else 0)
                batch_df = batch_df.drop(columns=['data'])

            # Join with metadata
            refined_shard = pd.merge(batch_df, df_meta, on='field_id', how='inner')
            
            if not refined_shard.empty:
                final_shards.append(refined_shard)
            
            # Explicit garbage collection to prevent RAM spikes
            del batch_df
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error processing batch {key}: {e}")

    # 3. Final Assembly and Export
    if not final_shards:
        raise ValueError("Silver Layer is empty. Check field_id joins.")

    logger.info("Concatenating all shards into final Silver DataFrame...")
    df_silver = pd.concat(final_shards, ignore_index=True)
    
    # Clean up shards list from memory
    del final_shards
    gc.collect()

    # --- SAVE TO MINIO ---
    logger.info(f"Saving refined data to {silver_bucket}...")
    buffer = io.BytesIO()
    df_silver.to_parquet(buffer, index=False)
    
    s3.put_object(
        Bucket=silver_bucket, 
        Key='refined_agriculture_data.parquet', 
        Body=buffer.getvalue()
    )
    
    logger.info("Silver Layer successfully processed and saved!")
    return df_silver

@pipeline
def agriculture_silver_pipeline():
    silver_refinery_engine()

if __name__ == "__main__":
    agriculture_silver_pipeline()