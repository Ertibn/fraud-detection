"""SHAP-based model explainability for fraud detection."""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class SHAPExplainer:
    """SHAP-based model interpretation."""

    def __init__(self, model, X_train, model_type="tree"):
        """Initialize SHAP explainer."""
        self.model = model
        self.X_train = X_train
        self.model_type = model_type

        # Create SHAP explainer based on model type
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_train)

    def calculate_shap_values(self, X):
        """Calculate SHAP values for dataset."""
        shap_values = self.explainer.shap_values(X)

        # For binary classification, take fraud class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    def plot_summary(self, X, feature_names=None, plot_type="bar", save_path=None):
        """Plot SHAP summary plot."""
        shap_values = self.calculate_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "bar":
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        elif plot_type == "beeswarm":
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_force_plot(self, X, sample_idx, feature_names=None, save_path=None):
        """Plot SHAP force plot for single prediction."""
        shap_values = self.calculate_shap_values(X)

        fig = plt.figure()
        shap.force_plot(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value,
            shap_values[sample_idx],
            X.iloc[sample_idx] if hasattr(X, "iloc") else X[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_feature_importance(self, X, feature_names=None, top_n=10):
        """Get global feature importance from SHAP values."""
        shap_values = self.calculate_shap_values(X)

        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]

        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_n]

        importance_data = {
            "Feature": [feature_names[i] for i in indices],
            "Mean Absolute SHAP": [importance[i] for i in indices],
        }

        return pd.DataFrame(importance_data)

    def plot_dependence(self, X, feature_idx, feature_names=None, save_path=None):
        """Plot SHAP dependence plot for feature."""
        shap_values = self.calculate_shap_values(X)

        feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"

        fig, ax = plt.subplots(figsize=(10, 6))

        if hasattr(X, "iloc"):
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_names,
                show=False,
            )
        else:
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_names,
                show=False,
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_individual_prediction_explanation(self, X, sample_idx, feature_names=None):
        """Get SHAP explanation for individual prediction."""
        shap_values = self.calculate_shap_values(X)

        if hasattr(X, "iloc"):
            sample = X.iloc[sample_idx].values
        else:
            sample = X[sample_idx]

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(sample))]

        # Calculate contribution of each feature
        shap_contrib = shap_values[sample_idx]

        explanation = {
            "Base Value": self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value,
            "Features": [],
        }

        # Sort by absolute contribution
        indices = np.argsort(np.abs(shap_contrib))[::-1]

        for idx in indices:
            explanation["Features"].append(
                {
                    "Name": feature_names[idx],
                    "Value": sample[idx],
                    "SHAP": shap_contrib[idx],
                    "Impact": "Increases" if shap_contrib[idx] > 0 else "Decreases",
                }
            )

        return explanation

    @staticmethod
    def plot_feature_importance_comparison(models_importance, save_path=None):
        """Compare feature importance across models."""
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models_importance[0]))
        width = 0.25

        for i, (model_name, importance) in enumerate(models_importance.items()):
            ax.bar(x + i * width, importance, width, label=model_name)

        ax.set_ylabel("Importance Score")
        ax.set_title("Feature Importance Comparison Across Models")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
