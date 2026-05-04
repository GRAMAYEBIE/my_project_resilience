import joblib
import pandas as pd
import os
import mlflow  # <--- AJOUT
import mlflow.sklearn  # <--- AJOUT
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION MLFLOW ---
# Connexion au serveur Docker et création de l'expérience
mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("Agri_Resilience_Production")

# 1. Chargement et Nettoyage
print("📊 Loading data...")
df = pd.read_parquet("data_storage/raw/final_scoring.parquet") 

features = ['final_precipitation', 'ph_level', 'nitrogen_content', 'organic_matter']
target = 'loan_status'

for col in features:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=[target])
X = df[features]
y = df[target]

# Split avant tout
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- DÉBUT DU RUN MLFLOW (Encapsule tout l'entraînement) ---
with mlflow.start_run(run_name="Champion_Model_V1.3.2"):

    # 2. OVERSAMPLING
    print(f"⚖️ Balancing TRAIN set only...")
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Log des paramètres pour MLflow
    mlflow.log_param("method", "SMOTE")
    mlflow.log_param("k_neighbors", 1)

    # 3. Preprocessing
    print("⚙️ Training Scaler and Encoder...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_res)

    # 4. Training the Champion Model
    print("🧠 Training VotingClassifier...")
    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42))
        ],
        voting='soft'
    )
    model.fit(X_train_scaled, y_train_encoded)

    # --- ÉVALUATION & LOGS MLFLOW ---
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = le.transform(y_test)
    y_pred = model.predict(X_test_scaled)

    # Calcul des métriques avec gestion multiclasse
    acc = accuracy_score(y_test_encoded, y_pred)
    # On utilise 'weighted' car ta cible a plus de 2 classes
    prec = precision_score(y_test_encoded, y_pred, average='weighted')
    rec = recall_score(y_test_encoded, y_pred, average='weighted')
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')

    # Logs MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(model, "voting_model_champion")

    print("\n" + "⭐" + "="*45 + "⭐")
    print(f"   AGRI-RESILIENCE ENGINE : PERFORMANCE REPORT")
    print("="*47)
    print(f" ✅ ACCURACY  : {acc * 100:.2f}%")
    print(f" 🎯 PRECISION : {prec * 100:.2f}% (weighted)")
    print(f" 🔍 RECALL    : {rec * 100:.2f}% (weighted)")
    print(f" ⚖️ F1-SCORE  : {f1 * 100:.2f}% (weighted)")
    print("="*47)
    # 4. Matrice de Confusion avec Seaborn
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    
    # On récupère les noms réels des classes depuis le LabelEncoder
    # Exemple: ['Solvable', 'High-Risk'] ou ['Class 0', 'Class 1', 'Class 2']
    class_names = [f"Class {c}" for c in le.classes_] 
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Greens', 
        xticklabels=[f"Pred {name}" for name in class_names],
        yticklabels=[f"Actual {name}" for name in class_names]
    )

    plt.title('Agri-Resilience: Confusion Matrix (Multiclass Support)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout() # Évite que les labels soient coupés
    plt.show()

    # 5. Export Artifacts (Ton système de forçage original)
    base_path = r"E:\mon_projet_resilience"
    storage_path = os.path.join(base_path, "model_storage")
    prod_artifacts_path = os.path.join(base_path, "production", "artifacts")
    
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(prod_artifacts_path, exist_ok=True)

    for folder in [storage_path, prod_artifacts_path]:
        joblib.dump(model, os.path.join(folder, 'voting_model_champion.joblib'))
        joblib.dump(scaler, os.path.join(folder, 'scaler.joblib'))
        joblib.dump(le, os.path.join(folder, 'label_encoder.joblib'))

    artifacts = {"model": model, "scaler": scaler, "label_encoder": le, "features": features}
    joblib.dump(artifacts, os.path.join(storage_path, 'preprocessor_blueprint.joblib'))

    # Sauvegarde des CSV
    processed_data_path = os.path.join(base_path, "data_storage", "processed")
    os.makedirs(processed_data_path, exist_ok=True)
    X_train_res.to_csv(os.path.join(processed_data_path, 'train_features.csv'), index=False)
    pd.Series(y_train_res).to_csv(os.path.join(processed_data_path, 'train_labels.csv'), index=False)

    print(f"\n✅ SYSTEM FORCED & MLFLOW TRACKED")