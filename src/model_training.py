"""Model training for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import pickle
from pathlib import Path


class FraudDetectionModels:
    """Train and manage fraud detection models."""

    def __init__(self, random_state=42):
        """Initialize model trainer."""
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}

    def get_baseline_model(self):
        """Get Logistic Regression baseline."""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )

    def get_random_forest_model(self, **kwargs):
        """Get Random Forest ensemble."""
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": self.random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    def get_xgboost_model(self, **kwargs):
        """Get XGBoost ensemble."""
        params = {
            "n_estimators": 100,
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "scale_pos_weight": 1,  # Adjust based on class ratio
            "n_jobs": -1,
            "verbosity": 0,
        }
        params.update(kwargs)
        return XGBClassifier(**params)

    def train_model(self, model, X_train, y_train, model_name=None):
        """Train a single model."""
        model.fit(X_train, y_train)
        if model_name:
            self.models[model_name] = model
        return model

    def cross_validate_model(self, model, X, y, cv=5, metrics=None):
        """Perform stratified cross-validation."""
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv_splitter,
            scoring=metrics,
            return_train_score=True,
            n_jobs=-1,
        )

        return cv_results

    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test, cv=5):
        """Train multiple models and return evaluation results."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            auc,
            precision_recall_curve,
        )

        results = []

        # 1. Logistic Regression (Baseline)
        print("Training Logistic Regression...")
        lr_model = self.get_baseline_model()
        self.train_model(lr_model, X_train, y_train, "logistic_regression")

        lr_pred = lr_model.predict(X_test)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, lr_pred_proba)
        lr_auc_pr = auc(recall_vals, precision_vals)

        results.append(
            {
                "Model": "Logistic Regression",
                "Accuracy": accuracy_score(y_test, lr_pred),
                "Precision": precision_score(y_test, lr_pred),
                "Recall": recall_score(y_test, lr_pred),
                "F1-Score": f1_score(y_test, lr_pred),
                "ROC-AUC": roc_auc_score(y_test, lr_pred_proba),
                "AUC-PR": lr_auc_pr,
            }
        )

        # 2. Random Forest
        print("Training Random Forest...")
        rf_model = self.get_random_forest_model()
        self.train_model(rf_model, X_train, y_train, "random_forest")

        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, rf_pred_proba)
        rf_auc_pr = auc(recall_vals, precision_vals)

        results.append(
            {
                "Model": "Random Forest",
                "Accuracy": accuracy_score(y_test, rf_pred),
                "Precision": precision_score(y_test, rf_pred),
                "Recall": recall_score(y_test, rf_pred),
                "F1-Score": f1_score(y_test, rf_pred),
                "ROC-AUC": roc_auc_score(y_test, rf_pred_proba),
                "AUC-PR": rf_auc_pr,
            }
        )

        # 3. XGBoost
        print("Training XGBoost...")
        xgb_model = self.get_xgboost_model()
        self.train_model(xgb_model, X_train, y_train, "xgboost")

        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, xgb_pred_proba)
        xgb_auc_pr = auc(recall_vals, precision_vals)

        results.append(
            {
                "Model": "XGBoost",
                "Accuracy": accuracy_score(y_test, xgb_pred),
                "Precision": precision_score(y_test, xgb_pred),
                "Recall": recall_score(y_test, xgb_pred),
                "F1-Score": f1_score(y_test, xgb_pred),
                "ROC-AUC": roc_auc_score(y_test, xgb_pred_proba),
                "AUC-PR": xgb_auc_pr,
            }
        )

        results_df = pd.DataFrame(results)
        return results_df, {
            "logistic_regression": (lr_pred, lr_pred_proba),
            "random_forest": (rf_pred, rf_pred_proba),
            "xgboost": (xgb_pred, xgb_pred_proba),
        }

    def save_model(self, model_name, filepath):
        """Save trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.models[model_name], f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath, model_name=None):
        """Load trained model from disk."""
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        if model_name:
            self.models[model_name] = model

        return model

    def get_feature_importance(self, model_name, feature_names=None, top_n=10):
        """Get feature importance from tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]

        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"Model {model_name} does not support feature importance")

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        importance_data = {
            "Feature": [feature_names[i] for i in indices],
            "Importance": [importances[i] for i in indices],
        }

        return pd.DataFrame(importance_data)
