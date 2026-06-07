"""IP address to country geolocation mapping."""

import pandas as pd
import numpy as np
import ipaddress


class IPGeolocation:
    """Map IP addresses to countries using range-based lookup."""

    def __init__(self, ip_mapping_df):
        """Initialize with IP-to-country mapping dataframe."""
        self.ip_df = ip_mapping_df.copy()
        self.ip_df = self.ip_df.sort_values("lower_bound_ip_address")

    def ip_to_integer(self, ip_str):
        """Convert IP address string to 32-bit integer."""
        try:
            if pd.isna(ip_str):
                return None
            return int(ipaddress.IPv4Address(ip_str))
        except (ValueError, TypeError):
            return None

    def get_country_from_ip(self, ip_str):
        """Get country for a single IP address."""
        ip_int = self.ip_to_integer(ip_str)
        if ip_int is None:
            return "Unknown"

        # Binary search for matching IP range
        matches = self.ip_df[
            (self.ip_df["lower_bound_ip_address"] <= ip_int)
            & (self.ip_df["upper_bound_ip_address"] >= ip_int)
        ]

        if len(matches) > 0:
            return matches.iloc[0]["country"]
        return "Unknown"

    def enrich_with_country(self, fraud_df):
        """Enrich fraud dataframe with country information from IP."""
        fraud_df = fraud_df.copy()
        fraud_df["ip_country"] = fraud_df["ip_address"].apply(self.get_country_from_ip)
        return fraud_df

    def get_fraud_rate_by_country(self, fraud_df):
        """Calculate fraud rate by country."""
        country_stats = (
            fraud_df.groupby("ip_country").agg(
                {
                    "class": ["sum", "count", "mean"],
                }
            )
        )
        country_stats.columns = ["fraud_count", "total", "fraud_rate"]
        country_stats = country_stats.sort_values("fraud_count", ascending=False)
        return country_stats
