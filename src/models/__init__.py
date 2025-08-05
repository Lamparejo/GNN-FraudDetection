"""
Graph Neural Network models for fraud detection.

This module implements different GNN architectures optimized
for fraud detection in heterogeneous graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, HeteroConv
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from loguru import logger


class BaseGNNModel(nn.Module, ABC):
    """
    Base class for GNN models.
    
    Defines the common interface for all GNN models
    used in the fraud detection system.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 **kwargs):
        """
        Initialize base model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes for classification
            num_layers: Number of GNN layers
            dropout: Dropout rate
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize common components
        self._build_model(**kwargs)
        
    @abstractmethod
    def _build_model(self, **kwargs):
        """Build model-specific architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get node embeddings without final classification.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch tensor (for GraphSAINT)
            
        Returns:
            Node embeddings
        """
        # Default implementation - can be overridden
        return self.forward(x, edge_index, batch)


class GraphSAGEModel(BaseGNNModel):
    """
    GraphSAGE implementation for heterogeneous graphs.
    
    GraphSAGE is efficient for inductive inference, learning
    a function to aggregate neighborhood features.
    """
    
    def _build_model(self, aggr: str = "mean", **kwargs):
        """
        Build GraphSAGE model.
        
        Args:
            aggr: Aggregation type ('mean', 'max', 'add')
        """
        self.aggr = aggr
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        
        # Primeira camada
        self.convs.append(
            SAGEConv(self.input_dim, self.hidden_dim, aggr=self.aggr)
        )
        
        # Camadas intermediárias
        for _ in range(self.num_layers - 2):
            self.convs.append(
                SAGEConv(self.hidden_dim, self.hidden_dim, aggr=self.aggr)
            )
        
        # Última camada
        if self.num_layers > 1:
            self.convs.append(
                SAGEConv(self.hidden_dim, self.hidden_dim, aggr=self.aggr)
            )
        
        # Normalização batch
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        logger.info(f"GraphSAGE construído: {self.num_layers} camadas, "
                   f"dim oculta: {self.hidden_dim}, agregação: {self.aggr}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do GraphSAGE.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            edge_index: Índices das arestas [2, num_edges]
            batch: Tensor de batch (não usado nesta implementação)
            
        Returns:
            Logits de classificação [num_nodes, num_classes]
        """
        # Propagação através das camadas GraphSAGE
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classificação final
        out = self.classifier(x)
        
        return out
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Obtém embeddings antes da classificação."""
        # Propagação através das camadas GraphSAGE
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATModel(BaseGNNModel):
    """
    Implementação do Graph Attention Network (GAT).
    
    GAT usa mecanismos de atenção para ponderar a importância
    de diferentes vizinhos na agregação de informações.
    """
    
    def _build_model(self, heads: int = 4, concat: bool = True, **kwargs):
        """
        Constrói o modelo GAT.
        
        Args:
            heads: Número de cabeças de atenção
            concat: Se concatenar ou fazer média das cabeças
        """
        self.heads = heads
        self.concat = concat
        
        # Ajustar dimensões para cabeças de atenção
        if self.concat:
            hidden_dim_per_head = self.hidden_dim // self.heads
        else:
            hidden_dim_per_head = self.hidden_dim
        
        # Camadas GAT
        self.convs = nn.ModuleList()
        
        # Primeira camada
        self.convs.append(
            GATConv(
                self.input_dim,
                hidden_dim_per_head,
                heads=self.heads,
                concat=self.concat,
                dropout=self.dropout
            )
        )
        
        # Camadas intermediárias
        input_dim_next = self.hidden_dim if self.concat else hidden_dim_per_head
        
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    input_dim_next,
                    hidden_dim_per_head,
                    heads=self.heads,
                    concat=self.concat,
                    dropout=self.dropout
                )
            )
        
        # Última camada (sempre com concat=False para saída uniforme)
        if self.num_layers > 1:
            self.convs.append(
                GATConv(
                    input_dim_next,
                    self.hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=self.dropout
                )
            )
        
        # Normalização
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        logger.info(f"GAT construído: {self.num_layers} camadas, "
                   f"dim oculta: {self.hidden_dim}, cabeças: {self.heads}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do GAT.
        
        Args:
            x: Features dos nós
            edge_index: Índices das arestas
            batch: Tensor de batch
            
        Returns:
            Logits de classificação
        """
        # Propagação através das camadas GAT
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # Normalização apenas se não for a última camada
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Aplicar normalização na última camada
        x = self.norms[-1](x)
        x = F.elu(x)
        
        # Classificação final
        out = self.classifier(x)
        
        return out


class HeteroGNNModel(nn.Module):
    """
    Modelo GNN para grafos heterogêneos.
    
    Utiliza HeteroConv para lidar com diferentes tipos de nós
    e arestas em um grafo heterogêneo.
    """
    
    def __init__(self,
                 metadata: tuple,
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 model_type: str = "SAGE"):
        """
        Inicializa o modelo heterogêneo.
        
        Args:
            metadata: Metadados do grafo (node_types, edge_types)
            hidden_dim: Dimensão das camadas ocultas
            num_classes: Número de classes
            num_layers: Número de camadas
            dropout: Taxa de dropout
            model_type: Tipo de convolução ('SAGE' ou 'GAT')
        """
        super().__init__()
        
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # Construir modelo
        self._build_hetero_model()
        
        logger.info(f"HeteroGNN construído: {model_type}, "
                   f"{num_layers} camadas, dim: {hidden_dim}")
    
    def _build_hetero_model(self):
        """Constrói o modelo heterogêneo."""
        # Camadas de convolução heterogênea
        self.convs = nn.ModuleList()
        
        for i in range(self.num_layers):
            conv_dict = {}
            
            # Definir convoluções para cada tipo de aresta
            for edge_type in self.metadata[1]:  # edge_types
                if self.model_type == "SAGE":
                    # SAGEConv para cada tipo de aresta
                    conv_dict[edge_type] = SAGEConv(-1, self.hidden_dim)
                        
                elif self.model_type == "GAT":
                    # GATConv para cada tipo de aresta
                    conv_dict[edge_type] = GATConv(-1, self.hidden_dim, heads=4, concat=False)
                else:
                    # Fallback to SAGE
                    conv_dict[edge_type] = SAGEConv(-1, self.hidden_dim)
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Classificador final (apenas para nós de transação)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass do modelo heterogêneo.
        
        Args:
            x_dict: Dicionário de features por tipo de nó
            edge_index_dict: Dicionário de arestas por tipo
            
        Returns:
            Logits para nós de transação
        """
        # Propagação através das camadas
        for i, conv in enumerate(self.convs):
            # Aplicar convolução
            try:
                x_dict_new = conv(x_dict, edge_index_dict)
                
                # Verificar se algum tipo de nó desapareceu
                missing_types = set(x_dict.keys()) - set(x_dict_new.keys())
                if missing_types:
                    # Manter os tipos de nó que não foram atualizados
                    for node_type in missing_types:
                        x_dict_new[node_type] = x_dict[node_type]
                
                x_dict = x_dict_new
                
            except Exception as e:
                logger.error(f"Erro durante convolução {i}: {e}")
                raise e
            
            if x_dict is None:
                raise ValueError(f"HeteroConv {i} returned None")
            
            # Aplicar ativação e dropout apenas em nós válidos
            for node_type in list(x_dict.keys()):
                if x_dict[node_type] is not None:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(
                    x_dict[node_type], 
                    p=self.dropout, 
                    training=self.training
                )
        
        # Classificação final apenas para nós de transação
        if 'transaction' not in x_dict:
            raise KeyError(f"'transaction' not found in final x_dict. Available keys: {list(x_dict.keys())}")
        
        transaction_embeddings = x_dict['transaction']
        out = self.classifier(transaction_embeddings)
        
        return out
    
    def get_embeddings(self, x_dict: Dict[str, torch.Tensor], 
                      edge_index_dict: Dict[tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Obtém embeddings de todos os tipos de nó."""
        # Propagação através das camadas
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            # Aplicar ativação e dropout
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(
                    x_dict[node_type], 
                    p=self.dropout, 
                    training=self.training
                )
        
        return x_dict


class ModelFactory:
    """
    Factory para criação de modelos GNN.
    
    Facilita a criação e configuração de diferentes
    tipos de modelos baseados em configurações.
    """
    
    @staticmethod
    def create_model(model_type: str,
                    input_dim: int,
                    hidden_dim: int = 256,
                    num_classes: int = 2,
                    num_layers: int = 3,
                    dropout: float = 0.3,
                    metadata: Optional[tuple] = None,
                    **kwargs) -> nn.Module:
        """
        Cria modelo baseado no tipo especificado.
        
        Args:
            model_type: Tipo do modelo ('GraphSAGE', 'GAT', 'HeteroGNN')
            input_dim: Dimensão de entrada
            hidden_dim: Dimensão oculta
            num_classes: Número de classes
            num_layers: Número de camadas
            dropout: Taxa de dropout
            metadata: Metadados para modelos heterogêneos
            **kwargs: Argumentos adicionais
            
        Returns:
            Modelo GNN inicializado
        """
        model_type = model_type.upper()
        
        if model_type == "GRAPHSAGE" or model_type == "SAGE":
            return GraphSAGEModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
        
        elif model_type == "GAT":
            return GATModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                **kwargs
            )
        
        elif model_type == "HETEROGNN" or model_type == "HETERO":
            if metadata is None:
                raise ValueError("Metadados necessários para modelo heterogêneo")
            
            # Extrair model_type_inner se fornecido
            inner_model_type = kwargs.pop('model_type_inner', 'SAGE')
            
            return HeteroGNNModel(
                metadata=metadata,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                model_type=inner_model_type,
                **kwargs
            )
        
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Retorna lista de modelos disponíveis."""
        return ["GraphSAGE", "GAT", "HeteroGNN"]


# Funções auxiliares para inicialização de pesos
def init_weights(m):
    """Inicializa pesos das camadas lineares."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def count_parameters(model: nn.Module) -> int:
    """Conta o número de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
