import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import paths

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train=None):
        # On entraîne le scaler sur les features
        self.scaler.fit(X_train)
        # On entraîne le LabelEncoder sur la cible (y)
        if y_train is not None:
            self.label_encoder.fit(y_train)
            print(f"🏷️ Classes détectées : {self.label_encoder.classes_}")

    def transform(self, X):
        return self.scaler.transform(X)

    def transform_target(self, y):
        return self.label_encoder.transform(y)

    def inverse_transform_target(self, y_pred):
        return self.label_encoder.inverse_transform(y_pred)

    def save(self):
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, paths.scaler_path)
        # On sauvegarde aussi l'encodeur pour Streamlit
        joblib.dump(self.label_encoder, paths.out_dir / "label_encoder.joblib")
        print(f"💾 Scaler et LabelEncoder sauvegardés.")