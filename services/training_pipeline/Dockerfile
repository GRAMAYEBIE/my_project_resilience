FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances (assure-toi que mlflow et imbalanced-learn sont dedans)
COPY services/training_pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# On copie le contenu du dossier service
COPY services/training_pipeline/ /app/

# On définit le PYTHONPATH pour que "import src" ou "import training_pipeline" fonctionne
ENV PYTHONPATH="/app:/app/src"

CMD ["python", "-m", "training_pipeline.run"]