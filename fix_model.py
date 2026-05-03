import joblib
import pandas as pd
import os
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Import pour l'équilibrage
from imblearn.over_sampling import RandomOverSampler 

# 1. Chargement et Nettoyage
print("📊 Loading data...")
df = pd.read_parquet("data_storage/raw/final_scoring.parquet") 

features = ['final_precipitation', 'ph_level', 'nitrogen_content', 'organic_matter']
target = 'loan_status'

# Imputation par la médiane
for col in features:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 2. OVERSAMPLING (The Balancing Act)
print(f"⚖️ Balancing classes... Original distribution: \n{y.value_counts()}")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
print(f"✅ New distribution: \n{pd.Series(y_resampled).value_counts()}")

# 3. Preprocessing (Fit on resampled data)
print("⚙️ Training Scaler and Encoder...")
scaler_fitted = StandardScaler()
X_scaled = scaler_fitted.fit_transform(X_resampled)

le_fitted = LabelEncoder()
y_encoded = le_fitted.fit_transform(y_resampled)

# 4. Training the Champion Model
print("🧠 Training VotingClassifier on balanced data...")
model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced')), # Plus de poids ici
        ('dt', DecisionTreeClassifier())
    ],
    voting='soft',
    weights=[1, 3, 1] 
)
model.fit(X_scaled, y_encoded)

# 5. Export Artifacts
artifacts = {
    "model": model,
    "scaler": scaler_fitted,
    "label_encoder": le_fitted,
    "feature_names": features
}

os.makedirs('model_storage', exist_ok=True)
joblib.dump(artifacts, 'model_storage/preprocessor_blueprint.joblib')
joblib.dump(model, 'model_storage/model.joblib')

print("🚀 SUCCESS: Balanced model (v1.3.2) is ready!")