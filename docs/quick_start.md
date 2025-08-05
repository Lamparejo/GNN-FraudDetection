# Quick Start Guide - Fraud Detection System

## Quick Start

### 1. Dependencies Installation

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Dataset

Download the **IEEE-CIS Fraud Detection Dataset** from Kaggle:
- https://www.kaggle.com/c/ieee-fraud-detection/data

Extract files to `ieee-fraud-detection/` folder:
```
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Verify Setup

```bash
python setup.py check
```

### 4. Run Quick Demo

```bash
python setup.py demo
```

### 5. Launch Dashboard

```bash
python setup.py dashboard
```

Access: http://localhost:8501

## 📋 Comandos Disponíveis

### Setup e Verificação
```bash
python setup.py check       # Verificar dependências e dados
python setup.py help        # Mostrar ajuda
```

### Execução
```bash
python setup.py demo        # Demo rápida (1000 amostras, 5 epochs)
python setup.py full        # Pipeline completo de produção
python setup.py dashboard   # Lançar interface web
```

### Execução Manual
```bash
# Treinamento customizado
python main.py --mode train --sample-size 10000 --config config.yaml

# Apenas avaliação
python main.py --mode evaluate --model-path results/

# Pipeline completo
python main.py --mode full --output-path results/
```

## ⚙️ Configuração

### Arquivo config.yaml

```yaml
# Exemplo de configuração personalizada
data:
  sample_size: 50000  # null para dataset completo
  test_split: 0.2
  
model:
  type: "GraphSAGE"  # "GAT", "HeteroGNN"
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 1024
  early_stopping_patience: 10
  
  class_weights:
    fraud: 10.0
    legitimate: 1.0
```

## 📊 Dashboard Features

### Páginas Disponíveis

1. **📊 Overview**
   - KPIs principais (Accuracy, Precision, Recall, F1)
   - Histórico de treinamento
   - Distribuição de fraudes

2. **📈 Métricas**
   - Matriz de confusão
   - Curvas ROC e Precision-Recall
   - Métricas por threshold

3. **🕸️ Análise de Grafos**
   - Visualização de redes de transações
   - Análise de centralidade
   - Detecção de comunidades

4. **🔍 Detecção em Tempo Real**
   - Simulação de transações
   - Alertas de fraude
   - Gauge de risco

5. **📋 Histórico**
   - Filtros por data, tipo, valor
   - Tabela de transações
   - Estatísticas

6. **⚙️ Configurações**
   - Parâmetros do modelo
   - Configurações de treinamento
   - Exportar/importar configs

## 🔧 Troubleshooting

### Problemas Comuns

#### Erro de CUDA
```bash
# Se não tiver GPU, forçar CPU
export CUDA_VISIBLE_DEVICES=""
```

#### Erro de Memória
```yaml
# Reduzir tamanho do batch no config.yaml
training:
  batch_size: 256  # Reduzir de 1024
```

#### Dataset não encontrado
```bash
# Verificar estrutura do diretório
ls ieee-fraud-detection/
# Deve mostrar os 4 arquivos CSV
```

#### Dependências
```bash
# Reinstalar PyTorch Geometric
pip uninstall torch-geometric
pip install torch-geometric
```

### Logs de Debug

```bash
# Habilitar logs detalhados
export LOG_LEVEL=DEBUG
python main.py --mode demo
```

## 📈 Performance

### Benchmarks de Referência

| Configuração | Dataset Size | Tempo Treinamento | AUC-ROC | Memória GPU |
|--------------|-------------|-------------------|---------|-------------|
| Demo         | 1K          | ~30s             | 0.85    | 1GB         |
| Small        | 10K         | ~2min            | 0.91    | 2GB         |
| Medium       | 100K        | ~15min           | 0.94    | 4GB         |
| Full         | 590K        | ~1h              | 0.96    | 8GB         |

### Otimizações

#### Para Desenvolvimento Rápido
```yaml
data:
  sample_size: 5000
training:
  epochs: 10
  batch_size: 512
```

#### Para Produção
```yaml
data:
  sample_size: null  # Dataset completo
training:
  epochs: 200
  batch_size: 2048
  mixed_precision: true
```

## 🐳 Docker (Opcional)

### Build
```bash
docker build -t fraud-detection .
```

### Run
```bash
docker run -p 8501:8501 -v $(pwd)/ieee-fraud-detection:/app/data fraud-detection
```

## 🚀 Deployment

### Modelo para Produção
```python
# Carregar modelo treinado
from main import FraudDetectionSystem

system = FraudDetectionSystem.load_system("results/production_model")

# Fazer predições
predictions, probabilities = system.predict(new_data)
```

### API REST (Exemplo)
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
system = FraudDetectionSystem.load_system("results/")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Processar dados e fazer predição
    result = system.predict(data)
    return jsonify({'fraud_probability': result})
```

## 📚 Recursos Adicionais

### Notebooks de Análise
```bash
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` - Análise exploratória
- `02_graph_construction.ipynb` - Construção de grafos  
- `03_model_analysis.ipynb` - Análise de modelos
- `04_interpretability.ipynb` - Explicabilidade

### Documentação Técnica
- `docs/architecture.md` - Arquitetura detalhada
- `docs/api.md` - Referência da API
- `docs/deployment.md` - Guia de deployment

### Testes
```bash
# Executar testes
python -m pytest tests/

# Cobertura
python -m pytest --cov=src tests/
```

## 📞 Suporte

### Issues Conhecidos
- Consulte: https://github.com/[seu-repo]/issues

### Contribuição
1. Fork o projeto
2. Crie feature branch
3. Commit alterações
4. Push para branch
5. Abra Pull Request

### Contato
- Email: [seu-email]
- LinkedIn: [seu-linkedin]
- GitHub: [seu-github]
