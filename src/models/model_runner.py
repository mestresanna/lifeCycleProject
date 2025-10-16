from sklearn.base import is_classifier

class ModelRunner:
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or model.__class__.__name__

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_feature_importances(self):
        """
        Returns a dictionary {feature_name: importance} if the underlying
        model supports feature_importances_ attribute (trees). Otherwise None.
        """
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
        # If wrapped in a pipeline, try to access last step
        elif hasattr(self.model, "named_steps"):
            last_step = list(self.model.named_steps.values())[-1]
            if hasattr(last_step, "feature_importances_"):
                return dict(zip(last_step.feature_names_in_, last_step.feature_importances_))
        return None
