"""
Data processing and graph construction module for fraud detection.

This module contains classes for loading, processing and transforming
tabular data into heterogeneous graphs for GNN training.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger

from ..utils import Timer


def _safe_hash(values):
    """
    Create a safe hash to use as ID.
    
    Args:
        values: Values to hash
        
    Returns:
        Safe hash as positive integer
    """
    import hashlib
    
    # Convert to string and create MD5 hash
    str_vals = tuple(str(v) for v in values)
    hash_obj = hashlib.md5(str(str_vals).encode())
    # Use only first 8 hex characters to avoid very large numbers
    hash_hex = hash_obj.hexdigest()[:8]
    # Convert to integer (max ~4 billion)
    return int(hash_hex, 16)


class DataLoader:
    """
    IEEE-CIS Fraud Detection Dataset loader.
    
    Manages loading and initial preprocessing of transaction
    and identity data.
    """
    
    def __init__(self, data_path: str = "ieee-fraud-detection"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.transaction_train = None
        self.transaction_test = None
        self.identity_train = None
        self.identity_test = None
        
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names replacing hyphens with underscores.
        
        Args:
            df: DataFrame with columns to be normalized
            
        Returns:
            DataFrame with normalized column names
        """
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.replace('-', '_')
        return df_copy
    
    def load_data(self, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load transaction and identity data.
        
        Args:
            sample_size: Number of samples to load (None = all)
            
        Returns:
            Tuple containing (combined_train_data, combined_test_data)
        """
        with Timer("Data loading"):
            # Load transaction data
            logger.info("Loading transaction data...")
            self.transaction_train = pd.read_csv(
                self.data_path / "train_transaction.csv",
                nrows=sample_size
            )
            self.transaction_train = self._normalize_column_names(self.transaction_train)
            
            self.transaction_test = pd.read_csv(
                self.data_path / "test_transaction.csv",
                nrows=sample_size
            )
            self.transaction_test = self._normalize_column_names(self.transaction_test)
            
            # Load identity data
            logger.info("Loading identity data...")
            self.identity_train = pd.read_csv(
                self.data_path / "train_identity.csv",
                nrows=sample_size
            )
            self.identity_train = self._normalize_column_names(self.identity_train)
            
            self.identity_test = pd.read_csv(
                self.data_path / "test_identity.csv",
                nrows=sample_size
            )
            self.identity_test = self._normalize_column_names(self.identity_test)
            
            # Merge data
            train_data = self._merge_data(self.transaction_train, self.identity_train)
            test_data = self._merge_data(self.transaction_test, self.identity_test)
            
            logger.info(f"Data loaded - Train: {len(train_data)}, Test: {len(test_data)}")
            
        return train_data, test_data
    
    def _merge_data(self, transaction_df: pd.DataFrame, 
                   identity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction and identity data.
        
        Args:
            transaction_df: Transaction DataFrame
            identity_df: Identity DataFrame
            
        Returns:
            Merged DataFrame
        """
        merged_df = transaction_df.merge(
            identity_df, 
            on='TransactionID', 
            how='left'
        )
        
        return merged_df
    
    def get_data_info(self) -> Dict[str, Any]:
        """Return information about loaded data."""
        if self.transaction_train is None:
            return {"status": "Data not loaded"}
        
        info = {
            "transaction_train_shape": self.transaction_train.shape if self.transaction_train is not None else None,
            "transaction_test_shape": self.transaction_test.shape if self.transaction_test is not None else None,
            "identity_train_shape": self.identity_train.shape if self.identity_train is not None else None,
            "identity_test_shape": self.identity_test.shape if self.identity_test is not None else None,
            "fraud_rate": self.transaction_train['isFraud'].mean() if self.transaction_train is not None else None,
            "missing_values": self.transaction_train.isnull().sum().sum() if self.transaction_train is not None else None
        }
        
        return info


class FeatureEngineer:
    """
    Feature engineer for transaction data.
    
    Responsible for creating new features, handling missing values
    and preparing data for graph construction.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame, 
                         is_train: bool = True) -> pd.DataFrame:
        """
        Apply feature engineering to DataFrame.
        
        Args:
            df: Input DataFrame
            is_train: If it's training set (for transformer fitting)
            
        Returns:
            DataFrame with processed features
        """
        with Timer("Feature engineering"):
            df = df.copy()
            
            # Clean data types first
            df = self._clean_data_types(df)
            
            # Temporal features
            df = self._create_temporal_features(df)
            
            # Aggregation features
            df = self._create_aggregation_features(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Encoding de variáveis categóricas
            df = self._encode_categorical_features(df, is_train)
            
            # Normalização de features numéricas
            df = self._scale_numerical_features(df, is_train)
            
            # Criação de IDs de entidades
            df = self._create_entity_ids(df)
            
            logger.info(f"Features processadas: {df.shape[1]} colunas")
            
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais."""
        # Converter TransactionDT para features temporais
        df['transaction_hour'] = (df['TransactionDT'] / 3600) % 24
        df['transaction_day'] = (df['TransactionDT'] / (3600 * 24)) % 7
        df['transaction_week'] = (df['TransactionDT'] / (3600 * 24 * 7)) % 52
        
        # Features de velocidade (se existirem D columns)
        d_cols = [col for col in df.columns if col.startswith('D')]
        for col in d_cols[:3]:  # Usar apenas as primeiras 3 para evitar complexidade excessiva
            if col in df.columns and df[col].notna().any():
                df[f'{col}_velocity'] = df['TransactionAmt'] / (df[col] + 1)
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de agregação."""
        # Agregações por cartão
        if 'card1' in df.columns:
            card_stats = df.groupby('card1')['TransactionAmt'].agg(['mean', 'std', 'count']).reset_index()
            card_stats.columns = ['card1', 'card1_amt_mean', 'card1_amt_std', 'card1_count']
            df = df.merge(card_stats, on='card1', how='left')
        
        # Agregações por email domain
        if 'P_emaildomain' in df.columns:
            email_stats = df.groupby('P_emaildomain')['TransactionAmt'].agg(['mean', 'count']).reset_index()
            email_stats.columns = ['P_emaildomain', 'email_amt_mean', 'email_count']
            df = df.merge(email_stats, on='P_emaildomain', how='left')
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores missing."""
        # Preencher valores categóricos com 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Preencher valores numéricos com mediana
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'isFraud':  # Não preencher o target
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, 
                                   is_train: bool) -> pd.DataFrame:
        """Codifica features categóricas."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'TransactionID']
        
        for col in categorical_cols:
            if is_train:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    # Lidar com valores não vistos no treino
                    known_values = set(self.encoders[col].classes_)
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_values else 'Unknown'
                    )
                    df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, 
                                is_train: bool) -> pd.DataFrame:
        """Normaliza features numéricas."""
        # Selecionar colunas numéricas para normalizar
        exclude_cols = ['TransactionID', 'isFraud'] + [
            col for col in df.columns if col.endswith('_id')
        ]
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if is_train:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                df[numerical_cols] = self.scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def _create_entity_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria IDs para diferentes entidades."""
        # User ID baseado em combinação de features
        user_features = []
        if 'addr1' in df.columns:
            user_features.append('addr1')
        if 'addr2' in df.columns:
            user_features.append('addr2')
        if 'P_emaildomain' in df.columns:
            user_features.append('P_emaildomain')
        
        if user_features:
            df['user_id'] = df[user_features].apply(
                lambda x: _safe_hash(x.astype(str)), axis=1
            )
        else:
            df['user_id'] = range(len(df))
        
        # Card ID
        if 'card1' in df.columns:
            df['card_id'] = df['card1']
        else:
            df['card_id'] = range(len(df))
        
        # Device ID baseado em device info
        device_features = ['DeviceType', 'DeviceInfo'] if 'DeviceType' in df.columns else []
        
        if device_features:
            available_features = [f for f in device_features if f in df.columns]
            if available_features:
                df['device_id'] = df[available_features].apply(
                    lambda x: _safe_hash(x.astype(str)), axis=1
                )
            else:
                df['device_id'] = range(len(df))
        else:
            df['device_id'] = range(len(df))
        
        return df

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza tipos de dados.
        
        Args:
            df: DataFrame com dados a serem limpos
            
        Returns:
            DataFrame com tipos limpos
        """
        df_clean = df.copy()
        
        # Converter colunas object para numeric onde possível
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Tentar converter para numérico
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception:
                    pass
        
        # Garantir que isFraud seja inteiro
        if 'isFraud' in df_clean.columns:
            df_clean['isFraud'] = df_clean['isFraud'].astype(int)
        
        # Preencher NaN com valores seguros
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'float32']:
                df_clean[col] = df_clean[col].fillna(0.0)
            elif df_clean[col].dtype in ['int64', 'int32']:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        
        return df_clean


class GraphBuilder:
    """
    Construtor de grafos heterogêneos para detecção de fraude.
    
    Transforma dados tabulares em grafos heterogêneos compatíveis
    com PyTorch Geometric.
    """
    
    def __init__(self):
        """Inicializa o construtor de grafos."""
        self.node_mappings = {}
        self.edge_mappings = {}
        
    def build_heterogeneous_graph(self, df: pd.DataFrame, 
                                target_col: str = 'isFraud') -> HeteroData:
        """
        Constrói grafo heterogêneo a partir do DataFrame.
        
        Args:
            df: DataFrame processado
            target_col: Nome da coluna target
            
        Returns:
            Objeto HeteroData do PyTorch Geometric
        """
        with Timer("Construção do grafo"):
            data = HeteroData()
            
            # Criar nós de transação
            data = self._create_transaction_nodes(data, df, target_col)
            
            # Criar nós de usuário
            data = self._create_user_nodes(data, df)
            
            # Criar nós de cartão
            data = self._create_card_nodes(data, df)
            
            # Criar nós de dispositivo
            data = self._create_device_nodes(data, df)
            
            # Criar arestas
            data = self._create_edges(data, df)
            
            logger.info(f"Grafo criado com {sum([data[node_type].num_nodes for node_type in data.node_types])} nós")
            logger.info(f"Tipos de nós: {data.node_types}")
            logger.info(f"Tipos de arestas: {data.edge_types}")
            
        return data
    
    def _create_transaction_nodes(self, data: HeteroData, df: pd.DataFrame, 
                                target_col: str) -> HeteroData:
        """Cria nós de transação."""
        # Selecionar features para nós de transação
        exclude_cols = ['TransactionID', 'user_id', 'card_id', 'device_id', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Garantir que todas as features são numéricas
        feature_df = df[feature_cols].copy()
        
        # Converter tudo para numérico, forçando erros a NaN e depois para 0
        for col in feature_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            feature_df[col] = feature_df[col].fillna(0.0)
        
        # Features dos nós
        node_features = torch.tensor(
            feature_df.values, dtype=torch.float
        )
        
        # Labels (apenas para nós de transação)
        if target_col in df.columns:
            labels = torch.tensor(df[target_col].values, dtype=torch.long)
            data['transaction'].y = labels
        
        data['transaction'].x = node_features
        data['transaction'].num_nodes = len(df)
        
        # Mapeamento de IDs
        self.node_mappings['transaction'] = dict(
            zip(df['TransactionID'], range(len(df)))
        )
        
        return data
    
    def _create_user_nodes(self, data: HeteroData, df: pd.DataFrame) -> HeteroData:
        """Cria nós de usuário."""
        # Obter usuários únicos
        unique_users = df['user_id'].unique()
        num_users = len(unique_users)
        
        # Features simples para usuários (pode ser expandido)
        user_features = torch.randn(num_users, 16)  # Features aleatórias por agora
        
        data['user'].x = user_features
        data['user'].num_nodes = num_users
        
        # Mapeamento de IDs
        self.node_mappings['user'] = dict(zip(unique_users, range(num_users)))
        
        return data
    
    def _create_card_nodes(self, data: HeteroData, df: pd.DataFrame) -> HeteroData:
        """Cria nós de cartão."""
        # Obter cartões únicos
        unique_cards = df['card_id'].unique()
        num_cards = len(unique_cards)
        
        # Features para cartões
        card_features = torch.randn(num_cards, 8)
        
        data['card'].x = card_features
        data['card'].num_nodes = num_cards
        
        # Mapeamento de IDs
        self.node_mappings['card'] = dict(zip(unique_cards, range(num_cards)))
        
        return data
    
    def _create_device_nodes(self, data: HeteroData, df: pd.DataFrame) -> HeteroData:
        """Cria nós de dispositivo."""
        # Obter dispositivos únicos
        unique_devices = df['device_id'].unique()
        num_devices = len(unique_devices)
        
        # Features para dispositivos
        device_features = torch.randn(num_devices, 8)
        
        data['device'].x = device_features
        data['device'].num_nodes = num_devices
        
        # Mapeamento de IDs
        self.node_mappings['device'] = dict(zip(unique_devices, range(num_devices)))
        
        return data
    
    def _create_edges(self, data: HeteroData, df: pd.DataFrame) -> HeteroData:
        """Cria arestas entre os nós."""
        # User -> Transaction
        user_transaction_edges = self._create_user_transaction_edges(df)
        data['user', 'makes', 'transaction'].edge_index = user_transaction_edges
        
        # Transaction -> Card
        transaction_card_edges = self._create_transaction_card_edges(df)
        data['transaction', 'uses', 'card'].edge_index = transaction_card_edges
        
        # Transaction -> Device
        transaction_device_edges = self._create_transaction_device_edges(df)
        data['transaction', 'from', 'device'].edge_index = transaction_device_edges
        
        return data
    
    def _create_user_transaction_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Cria arestas user -> transaction."""
        edges = []
        
        for _, row in df.iterrows():
            user_idx = self.node_mappings['user'][row['user_id']]
            transaction_idx = self.node_mappings['transaction'][row['TransactionID']]
            edges.append([user_idx, transaction_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_transaction_card_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Cria arestas transaction -> card."""
        edges = []
        
        for _, row in df.iterrows():
            transaction_idx = self.node_mappings['transaction'][row['TransactionID']]
            card_idx = self.node_mappings['card'][row['card_id']]
            edges.append([transaction_idx, card_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_transaction_device_edges(self, df: pd.DataFrame) -> torch.Tensor:
        """Cria arestas transaction -> device."""
        edges = []
        
        for _, row in df.iterrows():
            transaction_idx = self.node_mappings['transaction'][row['TransactionID']]
            device_idx = self.node_mappings['device'][row['device_id']]
            edges.append([transaction_idx, device_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class DataPipeline:
    """
    Complete data processing pipeline.
    
    Combines loading, feature engineering and graph construction
    into a single, reusable pipeline.
    """
    
    def __init__(self, data_path: str = "ieee-fraud-detection"):
        """
        Initialize data pipeline.
        
        Args:
            data_path: Path to data
        """
        self.data_path = data_path
        self.loader = DataLoader(data_path)
        self.engineer = FeatureEngineer()
        self.builder = GraphBuilder()
        
        self.train_data = None
        self.test_data = None
        self.train_graph = None
        self.test_graph = None
    
    def process_data(self, sample_size: Optional[int] = None) -> Tuple[HeteroData, HeteroData]:
        """
        Execute complete processing pipeline.
        
        Args:
            sample_size: Sample size for processing
            
        Returns:
            Tuple containing (train_graph, test_graph)
        """
        with Timer("Pipeline completo de dados"):
            # 1. Carregar dados
            train_raw, test_raw = self.loader.load_data(sample_size)
            
            # 2. Engenharia de features
            self.train_data = self.engineer.engineer_features(train_raw, is_train=True)
            self.test_data = self.engineer.engineer_features(test_raw, is_train=False)
            
            # 3. Construir grafos
            self.train_graph = self.builder.build_heterogeneous_graph(
                self.train_data, 'isFraud'
            )
            
            # Para teste, criar labels dummy se não existirem
            if 'isFraud' not in self.test_data.columns:
                self.test_data['isFraud'] = 0
            
            self.test_graph = self.builder.build_heterogeneous_graph(
                self.test_data, 'isFraud'
            )
            
            # 4. Add train/validation splits
            self.train_graph = self._add_train_val_split(self.train_graph)
        
        return self.train_graph, self.test_graph
    
    def _add_train_val_split(self, graph: HeteroData) -> HeteroData:
        """Add train/validation split to graph."""
        num_nodes = graph['transaction'].num_nodes
        
        # Simple split: 80% train, 20% validation
        train_size = int(0.8 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:train_size] = True
        val_mask[train_size:] = True
        
        graph['transaction'].train_mask = train_mask
        graph['transaction'].val_mask = val_mask
        
        return graph
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Return summary of processed data."""
        summary = {
            "loader_info": self.loader.get_data_info(),
        }
        
        if self.train_graph is not None:
            summary["train_graph"] = {
                "node_types": self.train_graph.node_types,
                "edge_types": self.train_graph.edge_types,
                "num_transaction_nodes": self.train_graph['transaction'].num_nodes,
                "num_features": self.train_graph['transaction'].x.shape[1],
                "fraud_rate": self.train_graph['transaction'].y.float().mean().item()
            }
        
        return summary
