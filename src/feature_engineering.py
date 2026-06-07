"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np
from datetime import datetime


class FraudFeatureEngineer:
    """Engineer features for fraud detection from e-commerce data."""

    def __init__(self):
        """Initialize feature engineer."""
        self.time_windows = [1, 24, 7]  # hours, hours, days

    def engineer_time_features(self, df):
        """Engineer time-based features from purchase_time."""
        df = df.copy()

        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["purchase_time"].dt.day
        df["month"] = df["purchase_time"].dt.month

        return df

    def engineer_time_since_signup(self, df):
        """Calculate time between signup and purchase (in hours)."""
        df = df.copy()
        df["time_since_signup_hours"] = (
            (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600
        )
        df["time_since_signup_days"] = (
            (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 86400
        )
        return df

    def engineer_transaction_velocity(self, df):
        """Engineer transaction frequency and velocity features per user."""
        df = df.copy()

        # Get transactions per user in different time windows
        for idx, row in df.iterrows():
            user_transactions = df[df["user_id"] == row["user_id"]]
            user_transactions_before = user_transactions[
                user_transactions["purchase_time"] < row["purchase_time"]
            ]

            # Recent transaction count (1 hour window)
            recent_1h = user_transactions_before[
                (row["purchase_time"] - user_transactions_before["purchase_time"]).dt.total_seconds()
                < 3600
            ]
            df.at[idx, "transactions_1h"] = len(recent_1h)

            # Recent transaction count (24 hour window)
            recent_24h = user_transactions_before[
                (row["purchase_time"] - user_transactions_before["purchase_time"]).dt.total_seconds()
                < 86400
            ]
            df.at[idx, "transactions_24h"] = len(recent_24h)

            # Recent transaction count (7 day window)
            recent_7d = user_transactions_before[
                (row["purchase_time"] - user_transactions_before["purchase_time"]).dt.total_seconds()
                < 604800
            ]
            df.at[idx, "transactions_7d"] = len(recent_7d)

        return df

    def engineer_device_features(self, df):
        """Engineer device-based features."""
        df = df.copy()

        # Device usage count (total transactions per device)
        device_counts = df["device_id"].value_counts()
        df["device_transaction_count"] = df["device_id"].map(device_counts)

        # Device fraud rate
        device_fraud = df.groupby("device_id")["class"].agg(["sum", "count"])
        device_fraud["fraud_rate"] = device_fraud["sum"] / device_fraud["count"]
        df["device_fraud_rate"] = df["device_id"].map(device_fraud["fraud_rate"])

        return df

    def engineer_categorical_features(self, df):
        """Engineer categorical variable features."""
        df = df.copy()

        # Source fraud rate
        source_fraud = df.groupby("source")["class"].agg(["sum", "count"])
        source_fraud["fraud_rate"] = source_fraud["sum"] / source_fraud["count"]
        df["source_fraud_rate"] = df["source"].map(source_fraud["fraud_rate"])

        # Browser fraud rate
        browser_fraud = df.groupby("browser")["class"].agg(["sum", "count"])
        browser_fraud["fraud_rate"] = browser_fraud["sum"] / browser_fraud["count"]
        df["browser_fraud_rate"] = df["browser"].map(browser_fraud["fraud_rate"])

        return df

    def engineer_user_behavior_features(self, df):
        """Engineer user behavior aggregation features."""
        df = df.copy()

        # User's average purchase value
        user_avg_purchase = df.groupby("user_id")["purchase_value"].mean()
        df["user_avg_purchase_value"] = df["user_id"].map(user_avg_purchase)

        # User's purchase deviation
        df["purchase_value_deviation"] = abs(
            df["purchase_value"] - df["user_avg_purchase_value"]
        )

        # User transaction count
        user_txn_count = df["user_id"].value_counts()
        df["user_transaction_count"] = df["user_id"].map(user_txn_count)

        # User fraud history
        user_fraud = df.groupby("user_id")["class"].agg(["sum", "count"])
        user_fraud["fraud_rate"] = user_fraud["sum"] / user_fraud["count"]
        df["user_fraud_rate"] = df["user_id"].map(user_fraud["fraud_rate"])

        return df

    def fit_engineer_features(self, df):
        """Fit and apply all feature engineering steps."""
        df = df.copy()

        df = self.engineer_time_features(df)
        df = self.engineer_time_since_signup(df)
        df = self.engineer_device_features(df)
        df = self.engineer_categorical_features(df)
        df = self.engineer_user_behavior_features(df)

        # Note: transaction velocity is expensive, applied selectively in notebooks
        return df


class CreditCardFeatureEngineer:
    """Engineer features for anonymized credit card data."""

    def engineer_time_features(self, df):
        """Engineer time-based features from Time column (seconds)."""
        df = df.copy()

        # Convert seconds to hours
        df["hour"] = (df["Time"] % 86400) / 3600
        df["hour_of_day"] = df["hour"].astype(int)

        # Time ranges
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)
        df["is_business_hours"] = (
            (df["hour_of_day"] >= 9) & (df["hour_of_day"] < 17)
        ).astype(int)

        return df

    def engineer_amount_features(self, df):
        """Engineer amount-based features."""
        df = df.copy()

        df["log_amount"] = np.log1p(df["Amount"])
        df["is_small_amount"] = (df["Amount"] < 25).astype(int)
        df["is_large_amount"] = (df["Amount"] > 500).astype(int)

        return df

    def fit_engineer_features(self, df):
        """Fit and apply all feature engineering steps."""
        df = df.copy()
        df = self.engineer_time_features(df)
        df = self.engineer_amount_features(df)
        return df
