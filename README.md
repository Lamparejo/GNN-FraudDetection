# Fraud Detection System using Graph Neural Networks

## Executive Summary

Financial fraud represents one of the most critical challenges facing modern payment systems, with global losses exceeding $48 billion annually. Traditional rule-based and machine learning approaches often fall short in detecting sophisticated fraud patterns that span across multiple interconnected entities. This project addresses this challenge by implementing a state-of-the-art Graph Neural Network (GNN) system that models payment transactions as a heterogeneous graph, enabling the detection of complex fraud patterns through network topology analysis.

## Business Problem & Impact

### The Challenge

Modern financial fraud has evolved beyond simple rule-based detection systems. Fraudsters now operate in coordinated networks, using shared devices, cards, and identities across multiple transactions. Traditional tabular machine learning approaches treat each transaction in isolation, missing critical relational patterns that could indicate fraudulent behavior.

Key challenges addressed:
- **Complex Fraud Networks**: Fraudsters use interconnected entities (cards, devices, emails) to hide their activities
- **Class Imbalance**: Fraudulent transactions represent <1% of all transactions, making detection difficult
- **Feature Engineering Limitations**: Traditional approaches miss graph-based patterns and network effects
- **Real-time Requirements**: Need for low-latency fraud detection in production environments
- **Interpretability**: Regulatory requirements demand explainable fraud detection decisions

### Business Value Delivered

1. **Improved Detection Accuracy**: Improvement in fraud detection rates compared to traditional ML approaches
2. **Reduced False Positives**: Graph-based features help distinguish legitimate from fraudulent patterns
3. **Network Effect Capture**: Ability to identify fraud rings and coordinated attacks
4. **Scalable Architecture**: Handles millions of transactions with sub-second inference
5. **Regulatory Compliance**: Built-in explainability features for audit and compliance requirements

## Technical Solution Architecture

### Graph Neural Network Approach

This system transforms tabular transaction data into a heterogeneous graph where:

- **Nodes** represent different entities:
  - Transactions (primary prediction targets)
  - Users/Cards (payment instruments)
  - Devices (transaction devices)

- **Edges** capture relationships:
  - User-Transaction: "makes transaction"
  - Transaction-Card: "uses payment card"
  - Transaction-Device: "originates from device"
  - Card-Card: "shared attributes"
  - Device-Device: "network connections"

### Model Architecture Justification

#### 1. GraphSAGE (Primary Model)
**Why chosen**: Optimized for inductive learning on large-scale graphs
- **Sampling Strategy**: Efficiently handles large transaction networks
- **Inductive Capability**: Can process new nodes without retraining
- **Aggregation Functions**: Mean, max, and LSTM aggregators capture different network patterns
- **Production Readiness**: Proven scalability in real-world fraud detection

#### 2. Graph Attention Networks (GAT)
**Why implemented**: Provides interpretable attention mechanisms
- **Attention Weights**: Shows which network connections influence fraud decisions
- **Multi-head Attention**: Captures different types of relationships simultaneously
- **Explainability**: Critical for regulatory compliance and model debugging

#### 3. Heterogeneous GNN (HeteroGNN)
**Why developed**: Handles multiple node and edge types natively
- **Type-specific Transformations**: Different neural networks for different entity types
- **Relation-specific Aggregation**: Separate handling of different relationship types
- **Semantic Preservation**: Maintains business meaning of different entity relationships

### Data Engineering Pipeline

#### Feature Engineering Strategy
1. **Temporal Features**: Hour, day, week patterns to capture fraud timing
2. **Aggregation Features**: Entity-level statistics (transaction frequency, amounts)
3. **Network Features**: Node degrees, clustering coefficients, centrality measures
4. **Risk Scores**: Historical fraud rates for entities
5. **Behavioral Features**: Deviation from historical patterns

#### Graph Construction Process
```python
# Example of heterogeneous graph construction
graph = HeteroData()
graph['transaction'].x = transaction_features
graph['user'].x = user_features
graph['card'].x = card_features

# Define relationships
graph['user', 'makes', 'transaction'].edge_index = user_transaction_edges
graph['transaction', 'uses', 'card'].edge_index = transaction_card_edges
```

## Key Features & Capabilities

### Core Functionality
- **Multi-model Architecture**: GraphSAGE, GAT, and HeteroGNN implementations
- **Real-time Inference**: Sub-second fraud scoring for live transactions
- **Batch Processing**: Efficient handling of historical data analysis

### Data Processing
- **Automated Feature Engineering**: 50+ engineered features from raw transaction data
- **Graph Construction**: Automated heterogeneous graph building from tabular data
- **Missing Value Handling**: Intelligent imputation preserving graph structure

### MLOps & Production
- **Configurable Architecture**: YAML-based configuration management
- **Model Versioning**: Automated model checkpointing and version control
- **Performance Monitoring**: Real-time metrics tracking and alerting
- **A/B Testing Framework**: Compare model performance across different strategies

### Visualization & Monitoring
- **Interactive Dashboard**: Streamlit-based real-time monitoring interface
- **Network Visualization**: Graph topology and fraud pattern visualization
- **Performance Metrics**: Comprehensive fraud detection KPIs
- **Explainability Views**: Model decision interpretation tools

## Installation & Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, recommended)
- 8GB RAM minimum (16GB recommended)

### Automated Setup (Recommended)

**For Linux/Mac:**
```bash
# Clone repository
git clone https://github.com/your-username/fraud-detection-gnn.git
cd fraud-detection-gnn

# Run automated setup script
chmod +x setup_environment.sh
./setup_environment.sh
```

**For Windows:**
```cmd
REM Clone repository
git clone https://github.com/your-username/fraud-detection-gnn.git
cd fraud-detection-gnn

REM Run automated setup script
setup_environment.bat
```

### Manual Installation

**Step 1: Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

**Step 2: Install Dependencies**
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install other requirements
pip install -r requirements.txt
```

### Data Setup - IEEE-CIS Fraud Detection Dataset

**Important**: The dataset files are excluded from Git (see `.gitignore`). You must download them manually.

**Step 1: Get Kaggle API Access**
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → Create New API Token
3. Download `kaggle.json` file

**Step 2: Configure Kaggle CLI**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials (Linux/Mac)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Setup credentials (Windows)
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\
```

**Step 3: Download Dataset**
```bash
# Method 1: Using Kaggle CLI (recommended)
kaggle competitions download -c ieee-fraud-detection

# Method 2: Manual download
# Visit: https://www.kaggle.com/c/ieee-fraud-detection/data
# Download all CSV files manually
```

**Step 4: Extract and Place Files**
```bash
# Extract downloaded files
unzip ieee-fraud-detection.zip

# Move files to project structure
mkdir -p data/raw
mv train_transaction.csv data/raw/
mv train_identity.csv data/raw/
mv test_transaction.csv data/raw/
mv test_identity.csv data/raw/
```

**Expected Data Structure:**
```
data/raw/
├── train_transaction.csv  (~651 MB)
├── train_identity.csv     (~25 MB)
├── test_transaction.csv   (~585 MB)
└── test_identity.csv      (~25 MB)
```

**Step 5: Verify Data Installation**
```bash
# Run comprehensive system diagnostic (recommended)
python diagnostic.py

# Manual file verification
ls -lh data/raw/
# Expected output:
# train_transaction.csv    ~651M
# train_identity.csv       ~25M  
# test_transaction.csv     ~585M
# test_identity.csv        ~25M

# Verify data loading works
python -c "
import pandas as pd
df = pd.read_csv('data/raw/train_transaction.csv', nrows=5)
print(f'Transaction data preview: {df.shape}')
print('Data files accessible')
"
```

**Troubleshooting**

If you encounter issues:

1. **Missing dependencies**: See `TROUBLESHOOTING.md` for detailed solutions
2. **Python version**: Ensure Python 3.8+ is installed
3. **PyTorch Geometric issues**: Try alternative installation methods in troubleshooting guide
4. **Data download problems**: Verify Kaggle credentials and competition acceptance
5. **Permission issues**: Check file permissions and directory access

**Quick Fix for Common Issues:**
```bash
# Run automated diagnostic and fix script
python diagnostic.py

# If errors persist, check troubleshooting guide
cat TROUBLESHOOTING.md
```

# Quick data validation
python -c "
import pandas as pd
import os

files = ['train_transaction.csv', 'train_identity.csv', 'test_transaction.csv', 'test_identity.csv']
for file in files:
    path = f'data/raw/{file}'
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=5)
        print(f'{file}: {df.shape} (showing first 5 rows)')
    else:
        print(f'ERROR: {file} not found')
"
```

> **Important**: The dataset is large (~1GB total). Ensure you have sufficient disk space and a stable internet connection. All data files are automatically excluded from git tracking via `.gitignore`.

### Quick Start
```bash
# Verify installation
python setup.py check

# Run quick demo with sample data
python setup.py demo

# Train full model
python main.py --mode full --sample-size 100000

# Launch interactive dashboard
streamlit run webapp/app.py
```

## Usage Tutorial

### 1. Basic Training
```bash
# Train with default configuration
python main.py --mode train

# Train with custom sample size
python main.py --mode train --sample-size 50000

# Train with custom configuration
python main.py --config custom_config.yaml --mode train
```

### 2. Model Evaluation
```bash
# Evaluate trained model
python main.py --mode evaluate --model-path results/models/

# Generate detailed evaluation report
python -c "
from main import FraudDetectionSystem
system = FraudDetectionSystem()
system.load_and_process_data()
system.create_model()
results = system.train_model()
print(results)
"
```

### 3. Real-time Inference
```python
from main import FraudDetectionSystem

# Load trained system
system = FraudDetectionSystem.load_system('results/models/')

# Make predictions on new data
predictions, probabilities = system.predict(new_transaction_data)

# Fraud probability for each transaction
fraud_scores = probabilities[:, 1]
```

### 4. Dashboard Usage
```bash
# Launch interactive dashboard
streamlit run webapp/app.py
```

Navigate to `http://localhost:8501` to access:
- Real-time fraud monitoring
- Model performance metrics
- Graph visualization tools
- Transaction analysis interface

## Configuration

### Model Configuration (`config.yaml`)
```yaml
model:
  type: "GraphSAGE"  # GraphSAGE, GAT, HeteroGNN
  hidden_dim: 256
  num_layers: 3
  dropout: 0.3
  
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 1024
  early_stopping_patience: 10
  
data:
  sample_size: 100000  # Use null for full dataset
  test_split: 0.2
  val_split: 0.1
```

### Hardware Configuration
```yaml
hardware:
  device: "auto"  # auto, cpu, cuda
  num_workers: 4
  mixed_precision: true
```

## Project Structure

```
fraud-detection-gnn/
├── src/                      # Core source code
│   ├── data/                 # Data processing pipeline
│   ├── models/               # GNN model implementations
│   ├── training/             # Training and evaluation
│   ├── utils/                # Utilities and configuration
│   ├── evaluation/           # Model evaluation tools
│   └── visualization/        # Visualization components
├── webapp/                   # Streamlit dashboard
├── notebooks/                # Jupyter analysis notebooks
│   ├── exploratory/          # Data exploration
│   └── modeling/             # Model development
├── docs/                     # Documentation
├── data/                     # Dataset directory
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed features
│   └── graphs/               # Constructed graphs
├── results/                  # Model outputs and results
│   └── models/               # Trained models and checkpoints
├── logs/                     # System logs
├── config.yaml               # Main configuration
├── requirements.txt          # Python dependencies
├── main.py                   # Main entry point
├── setup.py                  # Setup and validation script
├── install.py                # Installation script
├── README.md                 # This file
├── FEATURES.md               # Feature documentation
├── CHANGELOG.md              # Change log
└── LICENSE                   # MIT License
```

---

*Built with PyTorch Geometric, Streamlit, and modern MLOps practices for enterprise-grade fraud detection.*
