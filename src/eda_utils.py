"""Utility functions for exploratory data analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class EDAAnalyzer:
    """Perform exploratory data analysis on fraud datasets."""

    @staticmethod
    def get_class_imbalance_stats(df, target_col):
        """Get class imbalance statistics."""
        class_dist = df[target_col].value_counts()
        class_dist_norm = df[target_col].value_counts(normalize=True)

        stats = {
            "class_0_count": class_dist.get(0, 0),
            "class_1_count": class_dist.get(1, 0),
            "class_0_pct": class_dist_norm.get(0, 0) * 100,
            "class_1_pct": class_dist_norm.get(1, 0) * 100,
            "imbalance_ratio": class_dist.get(0, 1) / class_dist.get(1, 1),
        }
        return stats

    @staticmethod
    def get_numerical_statistics(df):
        """Get detailed numerical statistics."""
        return df.describe().T

    @staticmethod
    def get_categorical_statistics(df):
        """Get categorical feature statistics."""
        cat_cols = df.select_dtypes(include=["object"]).columns
        stats = {}
        for col in cat_cols:
            stats[col] = {
                "unique_values": df[col].nunique(),
                "top_value": df[col].value_counts().index[0],
                "top_value_count": df[col].value_counts().iloc[0],
                "missing": df[col].isnull().sum(),
            }
        return stats

    @staticmethod
    def analyze_fraud_by_categorical(df, categorical_col, target_col="class"):
        """Analyze fraud distribution across categorical feature."""
        analysis = df.groupby(categorical_col).agg(
            {
                target_col: ["sum", "count", "mean"],
            }
        )
        analysis.columns = ["fraud_count", "total", "fraud_rate"]
        analysis = analysis.sort_values("fraud_count", ascending=False)
        return analysis

    @staticmethod
    def analyze_fraud_by_numerical(df, numerical_col, target_col="class", bins=10):
        """Analyze fraud distribution across numerical feature (binned)."""
        df_copy = df.copy()
        df_copy["bins"] = pd.cut(df_copy[numerical_col], bins=bins)
        analysis = df_copy.groupby("bins").agg(
            {
                target_col: ["sum", "count", "mean"],
            }
        )
        analysis.columns = ["fraud_count", "total", "fraud_rate"]
        return analysis

    @staticmethod
    def get_correlation_with_target(df, target_col="class"):
        """Get correlation of features with target."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corrwith(df[target_col]).abs()
        correlations = correlations.sort_values(ascending=False)
        return correlations

    @staticmethod
    def detect_outliers_iqr(df, col, multiplier=1.5):
        """Detect outliers using IQR method."""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        return outliers

    @staticmethod
    def plot_class_distribution(df, target_col, title="Class Distribution"):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Count plot
        class_counts = df[target_col].value_counts()
        axes[0].bar(["Legitimate", "Fraud"], class_counts.values, color=["green", "red"])
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"{title} (Count)")
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v, str(v), ha="center", va="bottom")

        # Percentage plot
        class_pct = df[target_col].value_counts(normalize=True) * 100
        axes[1].pie(
            class_pct.values,
            labels=["Legitimate", "Fraud"],
            autopct="%1.1f%%",
            colors=["green", "red"],
        )
        axes[1].set_title(f"{title} (Percentage)")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_feature_fraud_comparison(df, feature, target_col="class", figsize=(12, 5)):
        """Plot fraud rate comparison across feature categories."""
        fraud_by_feature = df.groupby(feature)[target_col].agg(["sum", "count"])
        fraud_by_feature["fraud_rate"] = fraud_by_feature["sum"] / fraud_by_feature["count"]
        fraud_by_feature = fraud_by_feature.sort_values("fraud_rate", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Count plot
        axes[0].bar(range(len(fraud_by_feature)), fraud_by_feature["count"].values)
        axes[0].set_xticks(range(len(fraud_by_feature)))
        axes[0].set_xticklabels(fraud_by_feature.index, rotation=45, ha="right")
        axes[0].set_ylabel("Total Transactions")
        axes[0].set_title(f"Transaction Count by {feature}")

        # Fraud rate plot
        colors = ["red" if x > df[target_col].mean() else "green" for x in fraud_by_feature["fraud_rate"]]
        axes[1].bar(range(len(fraud_by_feature)), fraud_by_feature["fraud_rate"].values, color=colors)
        axes[1].axhline(y=df[target_col].mean(), color="blue", linestyle="--", label="Overall Fraud Rate")
        axes[1].set_xticks(range(len(fraud_by_feature)))
        axes[1].set_xticklabels(fraud_by_feature.index, rotation=45, ha="right")
        axes[1].set_ylabel("Fraud Rate")
        axes[1].set_title(f"Fraud Rate by {feature}")
        axes[1].legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_numerical_distribution(df, col, target_col="class", bins=30, figsize=(12, 4)):
        """Plot distribution of numerical feature split by fraud/legitimate."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        axes[0].hist(df[df[target_col] == 0][col], bins=bins, alpha=0.6, label="Legitimate", color="green")
        axes[0].hist(df[df[target_col] == 1][col], bins=bins, alpha=0.6, label="Fraud", color="red")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Distribution of {col}")
        axes[0].legend()

        # Box plot
        data_to_plot = [df[df[target_col] == 0][col], df[df[target_col] == 1][col]]
        axes[1].boxplot(data_to_plot, labels=["Legitimate", "Fraud"])
        axes[1].set_ylabel(col)
        axes[1].set_title(f"Box Plot of {col}")

        plt.tight_layout()
        return fig
