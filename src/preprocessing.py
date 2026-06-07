"""Data preprocessing and resampling for imbalanced data."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class FraudPreprocessor:
    """Preprocess fraud detection data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.scalers = {}
        self.encoders = {}

    def handle_missing_values(self, df, target_col="class"):
        """Handle missing values."""
        df = df.copy()

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Fill numerical missing values with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def remove_duplicates(self, df):
        """Remove duplicate rows."""
        initial_size = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_size - len(df)

        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

        return df

    def encode_categorical_features(self, df, categorical_cols, fit=True):
        """Encode categorical features."""
        df = df.copy()

        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col].astype(str))

        return df

    def scale_numerical_features(self, df, numerical_cols, fit=True):
        """Scale numerical features."""
        df = df.copy()

        if fit:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers["numerical"] = scaler
        else:
            if "numerical" in self.scalers:
                df[numerical_cols] = self.scalers["numerical"].transform(
                    df[numerical_cols]
                )

        return df

    def prepare_for_modeling(
        self, df, target_col, categorical_cols=None, numerical_cols=None, fit=True
    ):
        """Full preprocessing pipeline."""
        df = df.copy()

        # Step 1: Handle missing values
        df = self.handle_missing_values(df, target_col)

        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 3: Encode categorical features
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols, fit=fit)

        # Step 4: Scale numerical features
        if numerical_cols:
            df = self.scale_numerical_features(df, numerical_cols, fit=fit)

        return df


class ImbalancedDataHandler:
    """Handle class imbalance using SMOTE and undersampling."""

    def __init__(self, smote_ratio=0.5, random_state=42):
        """Initialize handler."""
        self.smote_ratio = smote_ratio
        self.random_state = random_state

    def apply_smote(self, X, y):
        """Apply SMOTE oversampling."""
        smote = SMOTE(
            sampling_strategy=self.smote_ratio,
            random_state=self.random_state,
            n_jobs=-1,
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        original_fraud_count = (y == 1).sum()
        new_fraud_count = (y_resampled == 1).sum()

        print(f"SMOTE Applied:")
        print(f"  Original fraud samples: {original_fraud_count}")
        print(f"  New fraud samples: {new_fraud_count}")
        print(f"  New fraud ratio: {new_fraud_count / len(y_resampled):.2%}")

        return X_resampled, y_resampled

    def apply_undersampling(self, X, y, sampling_strategy=0.5):
        """Apply random undersampling of majority class."""
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
        )
        X_resampled, y_resampled = undersampler.fit_resample(X, y)

        original_legitimate_count = (y == 0).sum()
        new_legitimate_count = (y_resampled == 0).sum()

        print(f"Undersampling Applied:")
        print(f"  Original legitimate samples: {original_legitimate_count}")
        print(f"  New legitimate samples: {new_legitimate_count}")
        print(f"  New fraud ratio: {(y_resampled == 1).sum() / len(y_resampled):.2%}")

        return X_resampled, y_resampled

    def apply_combined_resampling(self, X, y, smote_ratio=0.5, undersampling_ratio=0.5):
        """Apply combined SMOTE + undersampling."""
        pipeline = ImbPipeline(
            [
                ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=self.random_state)),
                (
                    "undersampler",
                    RandomUnderSampler(
                        sampling_strategy=undersampling_ratio, random_state=self.random_state
                    ),
                ),
            ]
        )
        X_resampled, y_resampled = pipeline.fit_resample(X, y)

        print(f"Combined SMOTE + Undersampling Applied:")
        print(f"  Final fraud ratio: {(y_resampled == 1).sum() / len(y_resampled):.2%}")

        return X_resampled, y_resampled

    def get_class_distribution(self, y):
        """Get class distribution statistics."""
        unique, counts = np.unique(y, return_counts=True)
        distribution = {}
        for u, c in zip(unique, counts):
            distribution[f"class_{u}"] = c
        distribution["total"] = len(y)
        distribution["imbalance_ratio"] = counts[0] / counts[1]

        return distribution
