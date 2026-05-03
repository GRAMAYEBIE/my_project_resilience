FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système indispensables
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# On installe TOUT ici pour être certain
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    joblib \
    streamlit \
    pandas \
    requests \
    plotly \
    pyarrow \
    numpy \
    scikit-learn==1.5.0 \
    xgboost \
    pydantic

# Ajoute ces lignes juste avant le COPY . .
RUN pip install --upgrade pip
RUN pip install scikit-learn==1.3.2 pandas numpy joblib

# Copie du code (comme on a fait tout à l'heure)
COPY services/inference_api/inference_api/ .

ENV PYTHONPATH=/app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]