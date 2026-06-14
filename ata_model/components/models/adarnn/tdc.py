"""
Temporal Distribution Characterization (TDC) Module

TDC automatically discovers temporal domains in time series data
by maximizing distribution differences between periods.

This is the key innovation of AdaRNN - instead of manually defining
time periods, TDC finds optimal splits that maximize domain diversity.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from .loss_transfer import TransferLoss


class TemporalDistributionCharacterization:
    """
    TDC: Temporal Distribution Characterization
    
    Automatically splits time series into K periods that maximize
    distribution differences between periods.
    
    Algorithm:
    1. Divide time range into N candidate splits (e.g., 10 equal parts)
    2. Greedily select splits that maximize total distribution distance
    3. Return optimal period boundaries
    
    Args:
        num_domains: Number of domains/periods to create
        num_splits: Number of candidate split points (default: 10)
        distance_type: Type of distance metric ('coral', 'mmd', 'cosine')
        device: torch device
    """
    
    def __init__(
        self,
        num_domains: int = 2,
        num_splits: int = 10,
        distance_type: str = 'coral',
        device: str = 'cuda'
    ):
        self.num_domains = num_domains
        self.num_splits = num_splits
        self.distance_type = distance_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def characterize(
        self,
        features: np.ndarray,
        time_col: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int]]:
        """
        Find optimal temporal splits that maximize distribution diversity
        
        Args:
            features: Feature matrix (n_samples, n_features)
            time_col: Optional time column for ordering (if not provided, assumes ordered)
            
        Returns:
            List of (start_idx, end_idx) tuples for each domain
        """
        n_samples = features.shape[0]
        
        # Convert to tensor
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        # Initialize selected boundaries
        selected = [0, self.num_splits]  # Start and end boundaries
        candidates = list(range(1, self.num_splits))  # Candidate split points
        
        # Greedily select splits that maximize distribution distance
        while len(selected) - 2 < self.num_domains - 1 and candidates:
            best_candidate = None
            best_distance = -float('inf')
            
            for candidate in candidates:
                # Try adding this candidate
                temp_selected = sorted(selected + [candidate])
                
                # Calculate total pairwise distance
                total_dist = self._compute_total_distance(feat_tensor, temp_selected, n_samples)
                
                if total_dist > best_distance:
                    best_distance = total_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        # Sort selected boundaries
        selected = sorted(selected)
        
        # Convert split ratios to actual indices
        domain_boundaries = []
        for i in range(1, len(selected)):
            start_idx = int(selected[i-1] / self.num_splits * n_samples)
            end_idx = int(selected[i] / self.num_splits * n_samples)
            domain_boundaries.append((start_idx, end_idx))
        
        return domain_boundaries
    
    def _compute_total_distance(
        self,
        features: torch.Tensor,
        splits: List[int],
        n_samples: int
    ) -> float:
        """Compute total pairwise distribution distance between all periods"""
        total_dist = 0.0
        criterion = TransferLoss(loss_type=self.distance_type, input_dim=features.shape[1])
        
        # Get feature slices for each period
        periods = []
        for i in range(1, len(splits)):
            start = int(splits[i-1] / self.num_splits * n_samples)
            end = int(splits[i] / self.num_splits * n_samples)
            if start < end:
                periods.append(features[start:end])
        
        # Compute pairwise distances
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                try:
                    dist = criterion.compute(periods[i], periods[j])
                    total_dist += dist.item()
                except:
                    continue
        
        return total_dist
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        time_col: str = 'time_sec'
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame into domains based on TDC
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            time_col: Name of time column
            
        Returns:
            List of DataFrames, one per domain
        """
        # Sort by time if time column exists
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        
        # Get features
        features = df[feature_cols].values
        
        # Get domain boundaries
        boundaries = self.characterize(features)
        
        # Split DataFrame
        domain_dfs = []
        for start_idx, end_idx in boundaries:
            domain_df = df.iloc[start_idx:end_idx].copy()
            domain_dfs.append(domain_df)
        
        return domain_dfs


class ManualTemporalSplit:
    """
    Manual temporal split based on time column values
    Use when you know the time boundaries (e.g., by year, month)
    """
    
    def __init__(self, time_boundaries: List[float]):
        """
        Args:
            time_boundaries: List of time values to split on
                           e.g., [100, 200, 300] creates 4 periods
        """
        self.time_boundaries = sorted(time_boundaries)
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = 'time_sec'
    ) -> List[pd.DataFrame]:
        """Split DataFrame based on time boundaries"""
        domain_dfs = []
        
        boundaries = [df[time_col].min()] + self.time_boundaries + [df[time_col].max() + 1]
        
        for i in range(len(boundaries) - 1):
            mask = (df[time_col] >= boundaries[i]) & (df[time_col] < boundaries[i+1])
            domain_df = df[mask].copy()
            if len(domain_df) > 0:
                domain_dfs.append(domain_df)
        
        return domain_dfs


class TimeBasedDomainSplitter:
    """
    Split data into domains based on time for IoT attack detection
    
    Supports:
    - Automatic TDC-based splitting
    - Manual time boundary splitting
    - Quantile-based splitting
    """
    
    def __init__(
        self,
        method: str = 'tdc',
        num_domains: int = 3,
        time_col: str = 'time_sec',
        **kwargs
    ):
        """
        Args:
            method: 'tdc' (automatic) or 'manual' or 'quantile'
            num_domains: Number of domains to create
            time_col: Name of time column in data
            **kwargs: Additional arguments for specific methods
        """
        self.method = method
        self.num_domains = num_domains
        self.time_col = time_col
        self.kwargs = kwargs
    
    def split(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame into temporal domains
        
        Args:
            df: Input DataFrame
            feature_cols: Feature columns (required for TDC method)
            
        Returns:
            List of DataFrames, one per domain
        """
        if self.method == 'tdc':
            if feature_cols is None:
                raise ValueError("feature_cols required for TDC method")
            
            tdc = TemporalDistributionCharacterization(
                num_domains=self.num_domains,
                distance_type=self.kwargs.get('distance_type', 'coral')
            )
            return tdc.split_dataframe(df, feature_cols, self.time_col)
        
        elif self.method == 'manual':
            boundaries = self.kwargs.get('boundaries', [])
            splitter = ManualTemporalSplit(boundaries)
            return splitter.split_dataframe(df, self.time_col)
        
        elif self.method == 'quantile':
            return self._quantile_split(df)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _quantile_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split by time quantiles"""
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        quantiles = np.linspace(0, 1, self.num_domains + 1)
        boundaries = df[self.time_col].quantile(quantiles).values
        
        domain_dfs = []
        for i in range(len(boundaries) - 1):
            mask = (df[self.time_col] >= boundaries[i])
            if i < len(boundaries) - 2:
                mask = mask & (df[self.time_col] < boundaries[i+1])
            domain_df = df[mask].copy()
            if len(domain_df) > 0:
                domain_dfs.append(domain_df)
        
        return domain_dfs


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create synthetic data with temporal drift
    n_samples = 1000
    n_features = 10
    
    # Period 1: Normal distribution centered at 0
    period1 = np.random.randn(n_samples // 3, n_features)
    
    # Period 2: Shifted distribution
    period2 = np.random.randn(n_samples // 3, n_features) + 2
    
    # Period 3: Different variance
    period3 = np.random.randn(n_samples // 3, n_features) * 3
    
    features = np.vstack([period1, period2, period3])
    
    # Test TDC
    tdc = TemporalDistributionCharacterization(num_domains=3, distance_type='coral')
    boundaries = tdc.characterize(features)
    
    print("Discovered domain boundaries:")
    for i, (start, end) in enumerate(boundaries):
        print(f"  Domain {i+1}: samples {start} to {end}")
