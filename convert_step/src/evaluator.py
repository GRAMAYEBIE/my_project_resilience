import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import paths, train_params

class Evaluator:
    """
    Evaluate the performance of the Voting Classifier and save the evidence.
    """

    @staticmethod
    def evaluate(y_true, y_pred) -> dict:
        """
        Calculates, displays, and saves performance metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_text = classification_report(y_true, y_pred)

        print("=" * 50)
        print("📊 MODEL EVALUATION : RICE RESILIENCE")
        print("=" * 50)
        print(f"Fianl Accuracy  : {accuracy:.4f}")
        print("\nClassification Report :")
        print(report_text)

        # automatic saves
        metrics = {
            "accuracy": accuracy,
            "weighted_f1": report_dict['weighted avg']['f1-score']
        }
        
        # result in the artefact file
        with open(paths.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        with open(paths.report_path, "w") as f:
            f.write(report_text)
            
        print(f"\n✅ Métriques save : {paths.out_dir}")
        return metrics

    @staticmethod
    def show_sample_prediction(y_true, y_pred) -> None:
        """
        Displays a concrete example to verify the business logic.
        """
        # Mapping 
        label_map = {
            0: "High Risk",
            1: "Premium",
            2: "Standard"
        }

        true_label = y_true.iloc[0]
        pred_label = y_pred[0]

        print("\n" + "=" * 50)
        print("🧐 ÉCHANTILLON TEST ") 
        print("=" * 50)
        print(f"Model prediction : {label_map[pred_label]} (Classe {pred_label})")
        print(f"Réality (Ground Truth): {label_map[true_label]} (Classe {true_label})")
        
        if pred_label == true_label:
            print("✨ Résult : CORRECT")
        else:
            print("⚠️ Résult : ERREUR")