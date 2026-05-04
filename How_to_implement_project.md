## Implementation & Deployment Guide: Agri-Resilience Ecosystem

### 1. Environment Setup

Bash

# Project isolation
python -m venv .venv
.\.venv\Scripts\activate  # Windows (PowerShell)
pip install -r requirements.txt

### 2. Infrastructure and Core Services

The project relies on a containerized architecture managed via Docker. If the WSL engine hangs, execute wsl --shutdown and restart Docker Desktop as an administrator.

Bash
docker-compose up -d

### 3. Data Pipeline (Medallion & Warehouse Architecture)

The raw data is already hosted on Minio. The processing flow is executed through a sequence of Python scripts to populate the warehouse:

Bronze_pipeline.py: Initial data ingestion from Minio.

Silver_pipeline.py: Data cleaning and normalization.

Gold_pipeline.py: Final feature engineering and aggregation.

Minio_to_postgres.py: Migration of processed data to the PostgreSQL relational warehouse.

Warehouse_actual.py: Final refresh of the production warehouse tables.

### 4. R&D Phase: From Notebook to Automation

Before industrialization, the business logic is validated in two stages:

Exploratory Notebook: Prototyping of machine learning models and statistical analysis.

Convert_step & Main.py: The convert_step module translates notebook logic into modular scripts, while main.py executes the entire src directory to generate the initial research artifacts.

### 5. Industrialization and Resilience (Production)

This stage bridges research and real-world deployment within the production directory:

run_pipeline.py train: The primary production engine. It orchestrates the full training cycle using real-time data sources to generate official model artifacts.

fix_model.py (The Resilience Pivot): This is the most critical actor for system stability. It intervenes after the production pipeline to ensure robustness by:

Applying SMOTE to handle class imbalance (Premium loans).

Training the VotingClassifier (The Champion Model), merging RandomForest, XGBoost, and Logistic Regression.

Forcing the injection of robust artifacts (85.57% accuracy) into deployment folders.

Registering the final version and metrics on MLflow for full auditability.

###6. Final MLOps Pipeline Deployment

Once the model is stabilized by the fix_model, the services are built and launched for the end-user:

Bash
docker-compose up --build feature_pipeline
docker-compose up --build training_pipeline
docker-compose up --build inference_pipeline

### 7. Service Monitoring Matrix


Streamlit UI	localhost:8050	User Interface & Control Tower (A/B/C Badge)

MLflow	localhost:5000	Experiment tracking and Model Registry

ZenML	localhost:8237	Pipeline orchestration visualization

FastAPI	localhost:8000	Real-time inference endpoint (NASA/OpenWeather integration)

Minio	localhost:9000	Object Storage (Bronze/Silver/Gold layers)

pgAdmin	localhost:5050	PostgreSQL Warehouse administration  

#### Strategic Conclusion

The success of the Agri-Resilience project lies in its hybrid architecture. By combining automated pipelines (run_pipeline.py) with a dedicated resilience module (fix_model.py), the system mitigates the risks of overfitting and data imbalance. This ensures that the financial credit decisions displayed on the Control Tower are not just statistically accurate, but architecturally sound and fully auditable via MLflow. This framework provides a scalable foundation for agricultural credit scoring in West Africa, bridging the gap between advanced data engineering and economic resilience.