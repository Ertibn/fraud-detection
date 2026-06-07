"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Load fraud detection datasets."""

    def __init__(self, data_path="data"):
        """Initialize data loader with path to data directory."""
        self.data_path = Path(data_path)

    def load_fraud_data(self):
        """Load e-commerce fraud dataset."""
        path = self.data_path / "Fraud_Data.csv"
        df = pd.read_csv(path)
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        return df

    def load_creditcard_data(self):
        """Load bank credit card dataset."""
        path = self.data_path / "creditcard.csv"
        df = pd.read_csv(path)
        return df

    def load_ip_mapping(self):
        """Load IP address to country mapping."""
        path = self.data_path / "IpAddress_to_Country.csv"
        df = pd.read_csv(path)
        return df

    def get_data_summary(self):
        """Get summary statistics for all datasets."""
        fraud_data = self.load_fraud_data()
        creditcard = self.load_creditcard_data()
        ip_mapping = self.load_ip_mapping()

        summary = {
            "fraud_data": {
                "shape": fraud_data.shape,
                "fraud_rate": fraud_data["class"].value_counts(normalize=True).to_dict(),
                "missing": fraud_data.isnull().sum().sum(),
                "duplicates": fraud_data.duplicated().sum(),
            },
            "creditcard": {
                "shape": creditcard.shape,
                "fraud_rate": creditcard["Class"].value_counts(normalize=True).to_dict(),
                "missing": creditcard.isnull().sum().sum(),
                "duplicates": creditcard.duplicated().sum(),
            },
            "ip_mapping": {
                "shape": ip_mapping.shape,
                "countries": ip_mapping["country"].nunique(),
                "missing": ip_mapping.isnull().sum().sum(),
            },
        }

        return summary
