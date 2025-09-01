"""
Training system for GNN fraud detection models.

This module implements the training, validation and
evaluation pipeline for Graph Neural Network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import numpy as np
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path
import time
import json
from loguru import logger
from tqdm import tqdm

from ..utils import Timer, device_manager


class FraudLoss(nn.Module):
    """
    Custom loss function for fraud detection.
    
    Combines CrossEntropy with weights to handle class imbalance
    and adds focal regularization.
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_focal: bool = True):
        """
        Initialize loss function.
        
        Args:
            class_weights: Weights to balance classes
            focal_alpha: Focal Loss alpha parameter
            focal_gamma: Focal Loss gamma parameter
            use_focal: Whether to use Focal Loss
        """
        super().__init__()
        
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        
        # Base loss function
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: True labels [N]
            
        Returns:
            Loss value
        """
        ce_loss = self.ce_loss(logits, targets)
        
        if not self.use_focal:
            return ce_loss.mean()
        
        # Focal Loss
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal component
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Apply alpha if specified
        if self.focal_alpha is not None:
            alpha_t = torch.where(
                targets == 1, 
                self.focal_alpha, 
                1 - self.focal_alpha
            )
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class MetricsCalculator:
    """Metrics calculator for model evaluation."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de classificação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições do modelo
            y_prob: Probabilidades preditas
            
        Returns:
            Dicionário com métricas
        """
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métricas específicas para fraude (classe 1)
        metrics['precision_fraud'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_fraud'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_fraud'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # AUC métricas
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Erro calculando AUC: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            
            # Especificidade
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics


class EarlyStopping:
    """Implementação de Early Stopping para treinamento."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        Inicializa Early Stopping.
        
        Args:
            patience: Número de épocas sem melhoria antes de parar
            min_delta: Mudança mínima para considerar como melhoria
            monitor: Métrica a ser monitorada
            mode: 'min' para diminuir, 'max' para aumentar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.is_better = (
            lambda current, best: current < best - self.min_delta
            if mode == 'min' 
            else lambda current, best: current > best + self.min_delta
        )
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Verifica se deve parar o treinamento.
        
        Args:
            metrics: Dicionário de métricas
            
        Returns:
            True se deve parar, False caso contrário
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Métrica {self.monitor} não encontrada")
            return False
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping ativado após {self.counter} épocas sem melhoria")
            
        return self.early_stop


class GNNTrainer:
    """
    Trainer principal para modelos GNN de detecção de fraude.
    
    Gerencia todo o processo de treinamento, validação e avaliação
    dos modelos de Graph Neural Networks.
    """
    
    def __init__(self,
                 model: nn.Module,
                 data: HeteroData,
                 config: Dict[str, Any],
                 save_dir: str = "results/models"):
        """
        Inicializa o trainer.
        
        Args:
            model: Modelo GNN a ser treinado
            data: Dados do grafo heterogêneo
            config: Configurações de treinamento
            save_dir: Diretório para salvar modelos
        """
        self.model = model.to(device_manager.device)
        
        # Mover dados para o dispositivo se possível
        if hasattr(data, 'to'):
            try:
                self.data = data.to(str(device_manager.device))
            except Exception:
                self.data = data
        else:
            self.data = data
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar otimizador
        self.optimizer = self._setup_optimizer()
        
        # Configurar scheduler
        self.scheduler = self._setup_scheduler()
        
        # Configurar função de perda
        self.criterion = self._setup_loss()
        
        # Configurar early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            monitor='val_f1',
            mode='max'
        )
        
        # Histórico de treinamento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Melhor modelo
        self.best_model_state = None
        self.best_val_score = 0.0
        
        logger.info(f"Trainer inicializado - Dispositivo: {device_manager.device}")
        logger.info(f"Modelo: {model.__class__.__name__}")
        
        # Inicializar parâmetros lazy para HeteroGNN
        if hasattr(model, 'convs') and hasattr(self.data, 'x_dict'):
            try:
                # Fazer um forward pass para inicializar parâmetros lazy
                logger.info("Inicializando parâmetros lazy do HeteroGNN...")
                model.train()
                with torch.no_grad():
                    _ = model(self.data.x_dict, self.data.edge_index_dict)
                logger.info("Parâmetros lazy inicializados com sucesso")
            except Exception as e:
                logger.warning(f"Não foi possível inicializar parâmetros lazy: {e}")
        
        # Contar parâmetros treináveis
        try:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Parâmetros treináveis: {total_params:,}")
        except ValueError as e:
            logger.warning(f"Não foi possível contar parâmetros: {e}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Configura o otimizador."""
        optimizer_config = self.config.get('optimizer', {})
        lr = float(self.config.get('learning_rate', 0.001))
        weight_decay = float(self.config.get('weight_decay', 5e-4))
        
        optimizer_type = optimizer_config.get('type', 'Adam')
        
        if optimizer_type == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Otimizador não suportado: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[Union[torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.StepLR]]:
        """Configura o scheduler de learning rate."""
        scheduler_config = self.config.get('scheduler', {})
        
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=float(scheduler_config.get('factor', 0.5)),
                patience=int(scheduler_config.get('patience', 5)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6))
            )
        elif scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 30)),
                gamma=float(scheduler_config.get('gamma', 0.1))
            )
        
        return None
    
    def _setup_loss(self) -> nn.Module:
        """Configura a função de perda."""
        class_weights = self.config.get('class_weights', {})
        
        if class_weights:
            weights = torch.tensor([
                class_weights.get('legitimate', 1.0),
                class_weights.get('fraud', 10.0)
            ], dtype=torch.float).to(device_manager.device)
        else:
            weights = None
        
        return FraudLoss(
            class_weights=weights,
            use_focal=self.config.get('use_focal_loss', True)
        )
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Executa uma época de treinamento.
        
        Returns:
            Tuple contendo (perda_média, métricas)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        try:
            # Forward pass
            if hasattr(self.data, 'x_dict'):  # Grafo heterogêneo
                logits = self.model(self.data.x_dict, self.data.edge_index_dict)
                labels = self.data['transaction'].y[self.data['transaction'].train_mask]
                train_logits = logits[self.data['transaction'].train_mask]
            else:  # Grafo homogêneo
                logits = self.model(self.data.x, self.data.edge_index)
                labels = self.data.y[self.data.train_mask]
                train_logits = logits[self.data.train_mask]
            
            # Garantir que labels são inteiros
            if labels.dtype != torch.long:
                labels = labels.long()
            
            # Calcular perda
            loss = self.criterion(train_logits, labels)
        except Exception as e:
            print(f"ERROR durante forward pass: {e}")
            print(f"Error type: {type(e)}")
            if hasattr(self.data, 'x_dict'):
                print(f"Available x_dict keys: {list(self.data.x_dict.keys())}")
            raise
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Calcular métricas
        with torch.no_grad():
            probs = torch.softmax(train_logits, dim=1)[:, 1]  # Probabilidade da classe fraude
            preds = train_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
        
        # Calcular métricas finais
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return total_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Executa uma época de validação.
        
        Returns:
            Tuple contendo (perda_média, métricas)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            # Forward pass
            if hasattr(self.data, 'x_dict'):  # Grafo heterogêneo
                logits = self.model(self.data.x_dict, self.data.edge_index_dict)
                labels = self.data['transaction'].y[self.data['transaction'].val_mask]
                val_logits = logits[self.data['transaction'].val_mask]
            else:  # Grafo homogêneo
                logits = self.model(self.data.x, self.data.edge_index)
                labels = self.data.y[self.data.val_mask]
                val_logits = logits[self.data.val_mask]
            
            # Calcular perda
            loss = self.criterion(val_logits, labels)
            
            # Calcular predições
            probs = torch.softmax(val_logits, dim=1)[:, 1]
            preds = val_logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
        
        # Calcular métricas
        metrics = MetricsCalculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return total_loss, metrics
    
    def train(self, epochs: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Executa o treinamento completo.
        
        Args:
            epochs: Número de épocas de treinamento
            verbose: Se exibir progresso
            
        Returns:
            Histórico de treinamento
        """
        logger.info(f"Iniciando treinamento para {epochs} épocas")
        
        with Timer(f"Treinamento completo ({epochs} épocas)"):
            
            progress_bar = tqdm(range(epochs), desc="Treinamento") if verbose else range(epochs)
            
            for epoch in progress_bar:
                # Época de treinamento
                train_loss, train_metrics = self.train_epoch()
                
                # Época de validação
                val_loss, val_metrics = self.validate_epoch()
                
                # Atualizar histórico
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                
                # Atualizar scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['f1'])
                    else:
                        self.scheduler.step()
                
                # Verificar melhor modelo
                current_val_score = val_metrics['f1']
                if current_val_score > self.best_val_score:
                    self.best_val_score = current_val_score
                    self.best_model_state = self.model.state_dict().copy()
                
                # Progress bar update
                if verbose and isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({
                        'train_loss': f"{train_loss:.4f}",
                        'val_f1': f"{val_metrics['f1']:.4f}",
                        'val_auc': f"{val_metrics['auc_roc']:.4f}"
                    })
                
                # Log a cada 10 épocas
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(
                        f"Época {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f} - "
                        f"Val F1: {val_metrics['f1']:.4f} - "
                        f"Val AUC: {val_metrics['auc_roc']:.4f}"
                    )
                
                # Early stopping
                val_metrics_with_loss = val_metrics.copy()
                val_metrics_with_loss['val_loss'] = val_loss
                
                if self.early_stopping(val_metrics_with_loss):
                    logger.info(f"Early stopping na época {epoch+1}")
                    break
        
        # Carregar melhor modelo
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Melhor modelo carregado (F1: {self.best_val_score:.4f})")
        
        # Salvar modelo final
        self.save_model()
        
        return self.history
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Salva o modelo treinado.
        
        Args:
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        if filename is None:
            filename = f"fraud_detection_model_{int(time.time())}.pt"
        
        model_path = self.save_dir / filename
        
        # Salvar modelo e metadados
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'config': self.config,
            'history': self.history,
            'best_val_score': self.best_val_score,
            'training_completed': True
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"Modelo salvo em {model_path}")
        
        # Salvar histórico em JSON
        history_path = model_path.with_suffix('.json')
        with open(history_path, 'w') as f:
            # Converter numpy arrays para listas para serialização JSON
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Converter valores numpy em dict de métricas
                        history_serializable[key] = [
                            {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
                            for metrics in value
                        ]
                    else:
                        history_serializable[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                else:
                    history_serializable[key] = value
            
            json.dump(history_serializable, f, indent=2)
        
        return model_path
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Retorna resumo do treinamento."""
        if not self.history['val_metrics']:
            return {"status": "Treinamento não concluído"}
        
        best_epoch = np.argmax([m['f1'] for m in self.history['val_metrics']])
        best_metrics = self.history['val_metrics'][best_epoch]
        
        summary = {
            "epochs_trained": len(self.history['train_loss']),
            "best_epoch": best_epoch + 1,
            "best_val_f1": best_metrics['f1'],
            "best_val_auc": best_metrics['auc_roc'],
            "final_train_loss": self.history['train_loss'][-1],
            "final_val_loss": self.history['val_loss'][-1],
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return summary
