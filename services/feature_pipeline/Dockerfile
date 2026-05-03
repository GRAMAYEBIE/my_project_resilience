# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for XGBoost, ZenML, and lightgbm
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy requirements first to leverage Docker cache
# Adjust the path to match your folder structure
COPY services/feature_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install MLOps core tools
RUN pip install "zenml[server]" s3fs pyarrow
# 3. Copy the shared source code (the heart of the project)
# This allows 'from src.data import DataManager' to work
# 3. Copy the shared source code (On garde le dossier parent 'production')
# On copie TOUT le dossier production dans /app/production
COPY production/ /app/production/

# 4. Copy the service-specific code
COPY services/feature_pipeline/ /app/feature_pipeline/

# 5. Initialize ZenML
RUN zenml init

# 6. PYTHONPATH (On pointe sur /app pour que 'production' soit un module)
ENV PYTHONPATH="/app:/app/production"
# Command
CMD ["python", "-m", "feature_pipeline.run"]