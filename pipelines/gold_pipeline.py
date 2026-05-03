import pandas as pd
import requests
import numpy as np
import s3fs
import logging
from zenml import step, pipeline

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MINIO CONFIG ---
S3_ENDPOINT = "http://localhost:9000"
S3_KEY = "admin"
S3_SECRET = "password123"

# Updated Bucket Name & Paths
SILVER_DATA_PATH = "s3://agriculture-silver/refined_agriculture_data.parquet"
GOLD_BUCKET_NAME = "agri-resillience-bucket"
GOLD_OUTPUT_PATH = f"s3://{GOLD_BUCKET_NAME}/final_scoring.parquet"

# Initialize FileSystem
s3 = s3fs.S3FileSystem(
    key=S3_KEY, 
    secret=S3_SECRET, 
    client_kwargs={'endpoint_url': S3_ENDPOINT}
)

# --- STEP 1: CLIMATE DATA (NASA + SILVER) ---
@step
def extract_climate_data() -> pd.DataFrame:
    """Fetches NASA API data and merges with Silver historical climate."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,GWETROOT&community=AG&longitude=-5.94&latitude=6.13&start=20240101&end=20241231&format=JSON"
    
    try:
        response = requests.get(url).json()
        nasa_precip = np.mean(list(response['properties']['parameter']['PRECTOTCORR'].values()))
    except Exception as e:
        logger.warning(f"NASA API connection failed, using fallback. Error: {e}")
        nasa_precip = 3.5  # Fallback based on typical local averages
        
    with s3.open(SILVER_DATA_PATH, mode='rb') as f:
        df_silver = pd.read_parquet(f)
    
    # Using 'field_id' as confirmed in your Silver layer
    # We use columns containing '_pr' for historical precipitation
    pr_cols = [c for c in df_silver.columns if '_pr' in c]
    df_silver['final_precipitation'] = (df_silver[pr_cols].mean(axis=1) * 0.7) + (nasa_precip * 0.3)
    
    return df_silver[['field_id', 'final_precipitation']]

# --- STEP 2: SOIL QUALITY ---
@step
def extract_soil_health() -> pd.DataFrame:
    """Extracts and imputes soil data from Silver layer."""
    with s3.open(SILVER_DATA_PATH, mode='rb') as f:
        df = pd.read_parquet(f)
    
    # Fill missing values (the 21k+ NaNs)
    df = df.fillna(df.mean(numeric_only=True))
    
    return pd.DataFrame({
        'field_id': df['field_id'],
        'ph_level': df['soil_phh2o_5-15cm_mean'] / 10,
        'nitrogen_content': df['soil_nitrogen_5-15cm_mean'],
        'organic_matter': df['soil_ocd_5-15cm_mean'] / 10
    })

# --- STEP 3: YIELD DATA ---
@step
def extract_historical_yields() -> pd.DataFrame:
    """Extracts historical yield from Silver."""
    with s3.open(SILVER_DATA_PATH, mode='rb') as f:
        df = pd.read_parquet(f)
    return df[['field_id', 'yield']].rename(columns={'yield': 'avg_yield'})

# --- STEP 4: GOLD ENGINE & STORAGE ---
@step
def gold_layer_decision_engine(
    climate: pd.DataFrame, 
    soil: pd.DataFrame, 
    yields: pd.DataFrame
) -> pd.DataFrame:
    """Calculates Credit Score using Min-Max normalization and saves to Gold Bucket."""
    
    # 1. Bucket Check (Sécurité MinIO)
    if not s3.exists(GOLD_BUCKET_NAME):
        s3.mkdir(GOLD_BUCKET_NAME)
        logger.info(f"Created bucket: {GOLD_BUCKET_NAME}")

    # 2. Merging data (2977 rows)
    df = climate.merge(soil, on='field_id').merge(yields, on='field_id')
    
    # 3. Normalisation (La clé pour éviter les scores bloqués à 100)
    # On transforme les valeurs brutes en ratios (0 à 1)
    yield_ratio = df['avg_yield'] / df['avg_yield'].max()
    precip_ratio = df['final_precipitation'] / df['final_precipitation'].max()
    
    # Pour le pH, on cherche l'idéal proche de 7 (neutre)
    # Ici on simplifie par un ratio de santé du sol
    ph_ratio = df['ph_level'] / df['ph_level'].max()
    
    # 4. Credit Scoring Logic (Pondération équilibrée)
    df['credit_score'] = (
        (yield_ratio * 40) +    # Performance historique (40%)
        (precip_ratio * 30) +   # Résilience climatique (30%)
        (ph_ratio * 30)         # Qualité du terrain (30%)
    ).round(2)
    
    # 5. Loan Decision Logic (Seuils adaptés aux ratios)
    # Avec la normalisation, un score > 80 est excellent.
    df['loan_status'] = np.where(df['credit_score'] > 80, 'PREMIUM_ELIGIBLE', 
                        np.where(df['credit_score'] > 50, 'STANDARD_ELIGIBLE', 'HIGH_RISK'))
    
    # 6. Final Save to MinIO
    with s3.open(GOLD_OUTPUT_PATH, mode='wb') as f:
        df.to_parquet(f)
    
    # Audit simple dans les logs
    mean_score = df['credit_score'].mean()
    logger.info(f"SUCCESS: Gold Layer saved. Average Score: {mean_score:.2f}")
    return df

# --- ORCHESTRATION ---
@pipeline
def agritech_scoring_pipeline():
    climate = extract_climate_data()
    soil = extract_soil_health()
    yields = extract_historical_yields()
    
    gold_layer_decision_engine(climate, soil, yields)

if __name__ == "__main__":
    agritech_scoring_pipeline.with_options(enable_cache=False)()