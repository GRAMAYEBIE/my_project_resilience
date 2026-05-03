import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.config import paths

def run_pipeline():
    print("🚀 Démarrage du Pipeline de Résilience Agricole...")

    loader = DataLoader()
    
    # Stratégie d'ingestion intelligente
    try:
        print("尝试 d'accès aux données Cloud (MinIO)...")
        df = loader.load_from_s3()
        print("✅ Données chargées avec succès depuis MinIO (S3).")
    except Exception as e_s3:
        print(f"⚠️ MinIO indisponible : {e_s3}")
        try:
            print("Tentative d'accès à PostgreSQL...")
            df = loader.load_from_postgres()
            print("✅ Données chargées depuis PostgreSQL.")
        except Exception as e_db:
            print(f"⚠️ Erreur DB : {e_db}. Repli sur le stockage local.")
            df = loader.load_data() 

    # 2. Split data
    X_train, X_test, y_train, y_test = loader.split_data(df)

    # ... après le split_data ...
    
    # 3. Preprocessing
    preprocessor = Preprocessor()
    preprocessor.fit(X_train, y_train) # On passe y_train pour le LabelEncoder
    
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # ON ENCODE LA CIBLE POUR XGBOOST
    y_train_encoded = preprocessor.transform_target(y_train)
    y_test_encoded = preprocessor.transform_target(y_test)
    
    preprocessor.save()

    # 4. Entraînement
    trainer = ModelTrainer()
    trainer.train(X_train_scaled, y_train_encoded) # On passe la version encodée
    trainer.save()

    # 5. Évaluation
    y_pred_encoded = trainer.predict(X_test_scaled)
    # On décode pour avoir le texte dans le rapport
    y_pred = preprocessor.inverse_transform_target(y_pred_encoded)
    
    evaluator = Evaluator()
    evaluator.evaluate(y_test, y_pred) # y_test est déjà en texte

    print("\n✨ Pipeline terminé ! Tes artefacts sont prêts pour l'UI Streamlit.")

if __name__ == "__main__":
    run_pipeline()