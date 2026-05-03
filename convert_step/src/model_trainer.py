import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.config import paths, train_params

class ModelTrainer:
    def __init__(self):
        # On utilise k_neighbors=1 comme dans ton notebook pour la classe 'Premium'
        self.smote = SMOTE(random_state=train_params.random_state, k_neighbors=1)
        
        # On définit les 3 mousquetaires de ton notebook
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced', random_state=42
        )
        self.xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='mlogloss', random_state=42
        )
        self.lr = LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        )

        # On crée un VotingClassifier pour combiner leurs forces (optionnel, ou prend le champion)
        # Ici on va entraîner le Champion (XGBoost ou RF selon tes tests)
        self.model = VotingClassifier(
    estimators=[
        ('rf', self.rf),
        ('xgb', self.xgb),
        ('lr', self.lr)
    ],
    voting='soft' # 'soft' permet d'utiliser les probabilités, c'est plus précis
) 

    def train(self, X_train, y_train):
        print(f"⚖️ Application du SMOTE (k_neighbors=1) pour équilibrer les classes...")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        
        print(f"🏋️ Entraînement du modèle Champion ({type(self.model).__name__})...")
        self.model.fit(X_res, y_res)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self):
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, paths.model_path)
        print(f"🏆 Modèle champion sauvegardé dans : {paths.model_path}")