from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelRunner:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or model.__class__.__name__

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average='macro', zero_division=0),
            "recall": recall_score(y_test, preds, average='macro', zero_division=0),
            "f1": f1_score(y_test, preds, average='macro', zero_division=0)
        }
