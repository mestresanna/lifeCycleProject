import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, class_labels=None):
        # Accepts integer or string labels
        self.results = []
        self.class_labels = class_labels or []

    def evaluate(self, model, X_test, y_test, backtest_predictions=None):
        if backtest_predictions is not None:
            y_pred = backtest_predictions
        else:
            y_pred = model.predict(X_test)

        # Convert class_labels to strings for classification_report
        target_names = [str(c) for c in self.class_labels]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0) if len(self.class_labels) == 2 else precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0) if len(self.class_labels) == 2 else recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0) if len(self.class_labels) == 2 else f1_score(y_test, y_pred, average='macro', zero_division=0)

        self.results.append({
            'model': model.name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'y_true': y_test,
            'y_pred': y_pred
        })

        print(f"\nClassification report for {model.name}:\n")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    def get_results_dataframe(self):
        return pd.DataFrame([
            {
                'model': r['model'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for r in self.results
        ]).set_index("model").sort_values(by="accuracy", ascending=False)

    def plot_confusion_matrices(self):
        target_names = [str(c) for c in self.class_labels]
        for r in self.results:
            cm = confusion_matrix(r['y_true'], r['y_pred'], labels=self.class_labels)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f"{r['model']} – Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()

    def plot_metric_comparison(self):
        df = self.get_results_dataframe()
        df.plot(kind='bar', figsize=(10, 6))
        plt.title("Model Comparison (Accuracy, Precision, Recall, F1)")
        plt.ylabel("Score")
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
