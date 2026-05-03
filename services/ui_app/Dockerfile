FROM python:3.11-slim

WORKDIR /app

# Ajoute mlflow ici
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    requests \
    pyarrow \
    plotly \
    fastapi \
    mlflow

RUN pip install --upgrade pip && pip install scikit-learn==1.5.1

# Copie du code
COPY services/ui_app/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]