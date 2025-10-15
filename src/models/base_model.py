from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class WrappedModel:
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
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds)
        }
