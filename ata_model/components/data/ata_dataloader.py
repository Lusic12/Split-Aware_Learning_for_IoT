"""
Multi-domain dataloader utilities for ATA training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler


class IoTDataset(Dataset):
    """Basic IoT attack detection dataset"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiDomainIoTDataset:
    """
    Dataset that holds multiple temporal domains for ATA training.
    
    Usage:
        dataset = MultiDomainIoTDataset(df, feature_cols, label_col, num_domains=3)
        domain_loaders = dataset.get_domain_loaders(batch_size=64)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label',
        time_col: str = 'time_sec',
        num_domains: int = 3,
        load_mode: str = 'time_series',  # 'time_series' | 'random_shuffle'
        split_method: str = 'quantile',  # 'quantile', 'tdc', 'manual'
        manual_boundaries: Optional[List[float]] = None,
        normalize: bool = True,
        random_state: int = 42
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.time_col = time_col
        self.num_domains = num_domains
        self.load_mode = load_mode
        self.normalize = normalize
        self.random_state = random_state
        self.scaler = StandardScaler() if normalize else None

        # Prepare ordering based on loading mode.
        if self.load_mode == 'random_shuffle':
            df = df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        else:
            df = df.sort_values(time_col).reset_index(drop=True)
        
        # Split into domains
        self.domain_dfs = self._split_domains(df, split_method, manual_boundaries)
        
        # Fit scaler on all data
        if self.normalize:
            all_features = df[feature_cols].values
            self.scaler.fit(all_features)
    
    def _split_domains(
        self,
        df: pd.DataFrame,
        method: str,
        boundaries: Optional[List[float]]
    ) -> List[pd.DataFrame]:
        """Split DataFrame into temporal domains"""

        if self.load_mode == 'random_shuffle':
            return self._random_split(df)
        
        if method == 'quantile':
            return self._quantile_split(df)
        elif method == 'manual' and boundaries is not None:
            return self._manual_split(df, boundaries)
        elif method == 'tdc':
            from ata_model.components.models.adarnn.tdc import TemporalDistributionCharacterization
            tdc = TemporalDistributionCharacterization(
                num_domains=self.num_domains,
                distance_type='coral'
            )
            return tdc.split_dataframe(df, self.feature_cols, self.time_col)
        else:
            return self._quantile_split(df)

    def _random_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Randomly split DataFrame into domains with near-equal sizes"""
        n = len(df)
        if n == 0:
            return []

        indices = np.arange(n)
        chunks = np.array_split(indices, self.num_domains)
        domain_dfs = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            domain_dfs.append(df.iloc[chunk].copy())
        return domain_dfs
    
    def _quantile_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split by time quantiles"""
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
    
    def _manual_split(self, df: pd.DataFrame, boundaries: List[float]) -> List[pd.DataFrame]:
        """Split by manual time boundaries"""
        all_bounds = [df[self.time_col].min()] + boundaries + [df[self.time_col].max() + 1]
        
        domain_dfs = []
        for i in range(len(all_bounds) - 1):
            mask = (df[self.time_col] >= all_bounds[i]) & (df[self.time_col] < all_bounds[i+1])
            domain_df = df[mask].copy()
            if len(domain_df) > 0:
                domain_dfs.append(domain_df)
        
        return domain_dfs
    
    def get_domain_datasets(self) -> List[IoTDataset]:
        """Get list of datasets for each domain"""
        datasets = []
        
        for domain_df in self.domain_dfs:
            features = domain_df[self.feature_cols].values
            labels = domain_df[self.label_col].values
            
            if self.normalize:
                features = self.scaler.transform(features)
            
            datasets.append(IoTDataset(features, labels))
        
        return datasets
    
    def get_domain_loaders(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True
    ) -> List[DataLoader]:
        """Get dataloaders for each domain"""
        datasets = self.get_domain_datasets()
        
        loaders = []
        for dataset in datasets:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last
            )
            loaders.append(loader)
        
        return loaders
    
    def get_domain_info(self) -> Dict:
        """Get information about each domain"""
        info = {}
        for i, domain_df in enumerate(self.domain_dfs):
            time_min = domain_df[self.time_col].min()
            time_max = domain_df[self.time_col].max()
            
            label_counts = domain_df[self.label_col].value_counts().to_dict()
            
            info[f'domain_{i}'] = {
                'n_samples': len(domain_df),
                'time_range': (time_min, time_max),
                'label_distribution': label_counts
            }
        
        return info


def create_ata_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    feature_cols: List[str],
    label_col: str = 'label',
    time_col: str = 'time_sec',
    num_domains: int = 3,
    batch_size: int = 64,
    normalize: bool = True,
    split_method: str = 'quantile',
    load_mode: str = 'time_series',
    train_shuffle: bool = True,
    random_state: int = 42
) -> Tuple[List[DataLoader], DataLoader, DataLoader, Optional[StandardScaler]]:
    """
    Create dataloaders for ATA training.
    
    Returns:
        train_loaders: List of dataloaders for each domain
        val_loader: Validation dataloader
        test_loader: Test dataloader
        scaler: Fitted scaler (for inference)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Create multi-domain training dataset and fit scaler on train only.
    train_dataset = MultiDomainIoTDataset(
        df=train_df,
        feature_cols=feature_cols,
        label_col=label_col,
        time_col=time_col,
        num_domains=num_domains,
        load_mode=load_mode,
        split_method=split_method,
        normalize=normalize,
        random_state=random_state
    )
    
    train_loaders = train_dataset.get_domain_loaders(batch_size=batch_size, shuffle=train_shuffle)
    scaler = train_dataset.scaler
    
    # Create validation and test datasets (single domain each)
    def create_single_loader(df):
        features = df[feature_cols].values
        labels = df[label_col].values
        
        if normalize and scaler is not None:
            features = scaler.transform(features)
        
        dataset = IoTDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    val_loader = create_single_loader(val_df)
    test_loader = create_single_loader(test_df)
    
    # Print domain info
    print("\n=== Multi-Domain Training Setup ===")
    print(f"Load mode: {load_mode}")
    domain_info = train_dataset.get_domain_info()
    for name, info in domain_info.items():
        print(f"{name}: {info['n_samples']} samples, time range {info['time_range']}")
    print()
    
    return train_loaders, val_loader, test_loader, scaler


def get_ata_domain_pairs(num_domains: int) -> List[Tuple[int, int]]:
    """
    Get all pairs of domains for distribution matching
    
    For 3 domains: [(0,1), (0,2), (1,2)]
    """
    pairs = []
    for i in range(num_domains):
        for j in range(i + 1, num_domains):
            pairs.append((i, j))
    return pairs


if __name__ == "__main__":
    # Example usage
    import os
    
    # Assume we have data files
    data_dir = "../../data_real"
    
    # Define feature columns (all columns except time_sec and label)
    feature_cols = [
        'node_id', 'parent_id', 'rpl_ver', 'rpl_rank', 'dis_sent', 'dio_sent',
        'dao_sent', 'nbr_dis_rcv', 'nbr_dio_rcv', 'nbr_dao_ack_rcv', 
        'nbr_fwd_to_me', 'nbr_fwd_to_others', 'nbr_fwd_bcast', 'nbr_rpl_ctrl',
        'nbr_non_rpl_ctrl', 'nbr_rpl_ver_rcv', 'nbr_rpl_rank_rcv', 'nbr_fwd_rpl',
        'nbr_fwd_non_rpl', 'diff_rpl_rank', 'diff_rpl_ver', 'norm_rank_diff',
        'ctrl_to_data_ratio', 'non_rpl_to_rpl_ratio', 'rpl_fwd_ratio',
        'non_rpl_fwd_ratio', 'total_fwd_ratio'
    ]
    
    if os.path.exists(os.path.join(data_dir, 'train.csv')):
        train_loaders, val_loader, test_loader, scaler = create_ata_dataloaders(
            train_path=os.path.join(data_dir, 'train.csv'),
            val_path=os.path.join(data_dir, 'val.csv'),
            test_path=os.path.join(data_dir, 'test.csv'),
            feature_cols=feature_cols,
            num_domains=3,
            batch_size=64
        )
        
        print(f"Created {len(train_loaders)} domain loaders")
        for i, loader in enumerate(train_loaders):
            print(f"  Domain {i}: {len(loader)} batches")
