"""Model evaluation utilities for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate classification models on imbalanced data."""

    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(y_true, y_pred):
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, output_dict=True)

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate all evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # ROC-AUC requires probability predictions
        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

            # AUC-PR (Area Under Precision-Recall Curve)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["auc_pr"] = auc(recall_vals, precision_vals)

        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_pr_curves(y_true, y_pred_proba, title="ROC & Precision-Recall Curves"):
        """Plot ROC and Precision-Recall curves."""
        from sklearn.metrics import roc_curve

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = auc(recall_vals, precision_vals)
        axes[1].plot(
            recall_vals,
            precision_vals,
            color="darkgreen",
            lw=2,
            label=f"PR curve (AUC = {auc_pr:.3f})",
        )
        axes[1].axhline(y=y_true.sum() / len(y_true), color="navy", linestyle="--", label="Baseline")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_model_comparison(results_df, metric="f1"):
        """Plot model comparison across metrics."""
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results_df.index))
        width = 0.15

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "auc_pr"]

        for i, m in enumerate(metrics):
            if m in results_df.columns:
                ax.bar(x + i * width, results_df[m], width, label=m)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison Across Metrics")
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(results_df.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig


class CrossValidationEvaluator:
    """Evaluate models using cross-validation."""

    @staticmethod
    def summarize_cv_results(cv_results):
        """Summarize cross-validation results."""
        summary = {}

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if f"test_{metric}" in cv_results:
                scores = cv_results[f"test_{metric}"]
                summary[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "scores": scores,
                }

        return summary

    @staticmethod
    def plot_cv_results(cv_results, metric="f1"):
        """Plot cross-validation results."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if f"test_{metric}" in cv_results:
            scores = cv_results[f"test_{metric}"]
            folds = np.arange(1, len(scores) + 1)

            ax.plot(folds, scores, marker="o", linestyle="-", linewidth=2, markersize=8)
            ax.axhline(y=np.mean(scores), color="r", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
            ax.fill_between(
                folds,
                np.mean(scores) - np.std(scores),
                np.mean(scores) + np.std(scores),
                alpha=0.2,
            )

            ax.set_xlabel("Fold")
            ax.set_ylabel(f"{metric.upper()} Score")
            ax.set_title(f"Cross-Validation {metric.upper()} Across Folds")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xticks(folds)

        plt.tight_layout()
        return fig
