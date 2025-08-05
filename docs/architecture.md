# Technical Architecture - GNN Fraud Detection System

## Architecture Overview

The system was designed following a modular and scalable architecture, with clear separation of responsibilities and enterprise development patterns.

### Project Structure

```
FraudDetection/
├── src/                          # Main source code
│   ├── data/                     # Data pipeline and graph construction
│   ├── models/                   # GNN model implementations
│   ├── training/                 # Training and validation system
│   ├── utils/                    # Utilities and configurations
│   └── visualization/            # Visualization components
├── dashboard/                    # Streamlit interface
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit and integration tests
├── docs/                         # Technical documentation
├── results/                      # Trained models and results
├── ieee-fraud-detection/         # IEEE-CIS dataset
├── config.yaml                   # Main configurations
├── main.py                       # Main entry point
└── requirements.txt              # Dependencies

```

## Technical Components

### 1. Data Pipeline (`src/data/`)

#### `DataLoader`
- **Responsibility**: Loading IEEE-CIS datasets
- **Features**:
  - Efficient loading with sampling support
  - Automatic merge of transaction and identity data
  - Data integrity validation
  - Detailed statistics logging

#### `FeatureEngineer`
- **Responsibility**: Feature engineering and preprocessing
- **Functionalities**:
  - Temporal features (hour, day, week)
  - Entity aggregations (card, email, etc.)
  - Intelligent missing value handling
  - Categorical variable encoding
  - Robust normalization
  - Entity ID creation

#### `GraphBuilder`
- **Responsabilidade**: Construção de grafos heterogêneos
- **Características**:
  - Suporte a múltiplos tipos de nós (Transaction, User, Card, Device)
  - Criação eficiente de arestas
  - Mapeamento de IDs para compatibilidade PyTorch Geometric
  - Validação de estrutura do grafo

### 2. Modelos GNN (`src/models/`)

#### Arquitetura Base (`BaseGNNModel`)
```python
class BaseGNNModel(nn.Module, ABC):
    """Classe abstrata definindo interface comum para GNNs"""
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """Constrói arquitetura específica"""
        pass
    
    @abstractmethod
    def forward(self, x, edge_index, batch=None):
        """Forward pass"""
        pass
```

#### Implementações Específicas

**GraphSAGE**
- Agregação de vizinhança escalável
- Suporte a inferência indutiva
- Múltiplos tipos de agregação (mean, max, add)
- Normalização batch entre camadas

**GAT (Graph Attention Network)**
- Mecanismo de atenção multi-head
- Ponderação automática de vizinhos
- Dropout de atenção para regularização
- Normalização layer-wise

**HeteroGNN**
- Suporte nativo a grafos heterogêneos
- Convoluções específicas por tipo de aresta
- Agregação inter-tipo de nó
- Flexibilidade para diferentes metadados

#### Factory Pattern
```python
model = ModelFactory.create_model(
    model_type="GraphSAGE",
    input_dim=128,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3
)
```

### 3. Sistema de Treinamento (`src/training/`)

#### `FraudLoss` - Função de Perda Customizada
```python
class FraudLoss(nn.Module):
    """
    Combina CrossEntropy com Focal Loss para desbalanceamento
    """
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        # Implementação otimizada para detecção de fraude
```

**Features**:
- Weighted CrossEntropy para balanceamento de classes
- Focal Loss para casos difíceis
- Suporte a class weights dinâmicos

#### `GNNTrainer` - Sistema de Treinamento
```python
class GNNTrainer:
    """Trainer completo com todas as funcionalidades empresariais"""
    
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self._setup_components()
```

**Funcionalidades Avançadas**:
- Early stopping com múltiplas métricas
- Learning rate scheduling
- Gradient clipping
- Mixed precision training (opcional)
- Checkpoint automático
- Logging estruturado
- Validação contínua

#### Métricas e Avaliação
```python
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob)
        }
```

### 4. Utilitários (`src/utils/`)

#### Sistema de Configuração
```python
class Config:
    """Gerenciador centralizado de configurações"""
    
    def get(self, key: str, default=None):
        """Acesso via dot notation: config.get('model.hidden_dim')"""
        
    @property
    def model(self) -> Dict:
        """Configurações do modelo"""
        
    @property  
    def training(self) -> Dict:
        """Configurações de treinamento"""
```

#### Gerenciamento de Dispositivos
```python
class DeviceManager:
    """Gerenciamento inteligente de CPU/GPU"""
    
    def __init__(self, device="auto"):
        self.device = self._get_device(device)
        
    def to_device(self, tensor):
        return tensor.to(self.device)
```

#### Sistema de Logging
- Logging estruturado com Loguru
- Rotação automática de logs
- Diferentes níveis por componente
- Integração com Weights & Biases (opcional)

## 🚀 Pipeline de Execução

### 1. Inicialização do Sistema
```python
system = FraudDetectionSystem("config.yaml")
```

### 2. Processamento de Dados
```python
# Carregamento e processamento automático
system.load_and_process_data()

# Resultado: grafos heterogêneos otimizados
train_graph = system.train_graph  # HeteroData
test_graph = system.test_graph    # HeteroData
```

### 3. Criação e Treinamento do Modelo
```python
# Criação automática baseada em configuração
system.create_model()

# Treinamento com monitoramento completo
history = system.train_model()
```

### 4. Avaliação e Persistência
```python
# Avaliação no conjunto de teste
results = system.evaluate_model()

# Salvamento completo do sistema
system.save_system("results/production_model")
```

## 📊 Dashboard Streamlit

### Arquitetura do Dashboard

```python
class FraudDetectionDashboard:
    """Dashboard principal com arquitetura modular"""
    
    def __init__(self):
        self.setup_session_state()
        
    def render_sidebar(self):
        """Navegação e controles globais"""
        
    def render_*_page(self):
        """Páginas específicas modulares"""
```

### Páginas Implementadas

1. **📊 Overview**: Métricas principais e KPIs
2. **📈 Métricas**: Análise detalhada de performance
3. **🕸️ Análise de Grafos**: Visualização de redes
4. **🔍 Detecção em Tempo Real**: Monitoring live
5. **📋 Histórico**: Análise temporal
6. **⚙️ Configurações**: Gestão de parâmetros

### Componentes Visuais

#### Métricas Interativas
```python
st.metric(
    label="🎯 Accuracy",
    value=f"{accuracy:.2%}",
    delta="+1.2%"
)
```

#### Visualizações Avançadas
- Grafos 3D interativos com NetworkX + Plotly
- Curvas ROC/PR dinâmicas
- Matrizes de confusão em tempo real
- Gauges de risco personalizados

## 🔧 Configuração e Deployment

### Arquivo de Configuração (`config.yaml`)
```yaml
# Configuração hierárquica e flexível
project:
  name: "fraud_detection_gnn"
  version: "1.0.0"

data:
  train_transaction: "ieee-fraud-detection/train_transaction.csv"
  sample_size: 50000  # Para desenvolvimento

model:
  type: "GraphSAGE"
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
  class_weights:
    fraud: 10.0
    legitimate: 1.0
```

### Executáveis de Linha de Comando

#### Treinamento Completo
```bash
python main.py --mode full --config config.yaml --sample-size 10000
```

#### Treinamento Apenas
```bash
python main.py --mode train --sample-size 50000
```

#### Dashboard
```bash
streamlit run dashboard/app.py
```

### Variáveis de Ambiente
```bash
# Configuração de GPU
CUDA_VISIBLE_DEVICES=0,1

# Configuração de logging
LOG_LEVEL=INFO

# Configuração de dados
DATA_PATH=/path/to/ieee-fraud-detection/
```

## 🧪 Testes e Validação

### Estrutura de Testes
```
tests/
├── unit/                    # Testes unitários
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   └── test_training.py
├── integration/             # Testes de integração
│   ├── test_full_pipeline.py
│   └── test_dashboard.py
└── performance/             # Testes de performance
    ├── test_memory_usage.py
    └── test_inference_speed.py
```

### Validação de Modelos
- Cross-validation temporal
- Validação holdout
- Análise de drift de dados
- Teste de robustez

## 📈 Monitoramento e Observabilidade

### Métricas de Sistema
- Tempo de treinamento por época
- Uso de memória GPU/CPU
- Throughput de inferência
- Latência de predição

### Métricas de Negócio
- Taxa de detecção de fraude
- Taxa de falsos positivos
- Valor monetário protegido
- ROI do sistema

### Alertas Automáticos
- Degradação de performance
- Drift de dados
- Falhas de sistema
- Anomalias de uso

## 🔒 Segurança e Compliance

### Tratamento de Dados Sensíveis
- Anonymização automática
- Encryption em repouso
- Audit logs completos
- GDPR compliance ready

### Explicabilidade
- GNNExplainer integration
- SHAP values para features
- Análise de subgrafos importantes
- Relatórios de decisão auditáveis

## 🚀 Escalabilidade

### Otimizações de Performance
- Batch processing otimizado
- Memory mapping para grandes datasets
- Distributed training (multi-GPU)
- Model quantization para inferência

### Arquitetura Cloud-Ready
- Containerização com Docker
- Kubernetes deployment manifests
- CI/CD pipelines
- Auto-scaling baseado em carga

## 📚 Próximos Passos

### Roadmap Técnico
1. **Inferência em Streaming**: Apache Kafka integration
2. **AutoML**: Hyperparameter optimization automático
3. **Federated Learning**: Treinamento distribuído
4. **Edge Deployment**: Inferência em dispositivos móveis
5. **Advanced Explainability**: Counterfactual explanations

### Melhorias de Negócio
1. **A/B Testing Framework**: Comparação de modelos em produção
2. **Business Rules Engine**: Regras de negócio customizáveis
3. **Real-time Alerting**: Sistema de notificações avançado
4. **Customer Dashboard**: Interface para analistas de fraude
5. **API Gateway**: Exposição de APIs para integração
