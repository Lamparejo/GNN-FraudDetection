# Troubleshooting Guide

## System Diagnostic Issues Resolution

Based on the diagnostic results, here are step-by-step solutions for common issues:

### Python Version Issue

**Problem**: `Python 3.6 not supported (requires 3.8+)`

**Solutions**:

1. **Update Python (Recommended)**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.8 python3.8-pip python3.8-venv
   
   # CentOS/RHEL
   sudo yum install python38 python38-pip
   
   # macOS
   brew install python@3.8
   
   # Windows
   # Download from https://www.python.org/downloads/
   ```

2. **Create Virtual Environment**:
   ```bash
   python3.8 -m venv fraud_detection_env
   source fraud_detection_env/bin/activate  # Linux/Mac
   # or
   fraud_detection_env\Scripts\activate     # Windows
   ```

### Missing Dependencies

**Problem**: Missing PyTorch Geometric, NetworkX, and Loguru

**Solution**:

```bash
# First, ensure you have the correct Python version activated
python --version  # Should show 3.8+

# Install missing packages in correct order
pip install --upgrade pip

# Install PyTorch first (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install other missing packages
pip install networkx loguru

# Verify installation
python -c "import torch_geometric; print('PyTorch Geometric:', torch_geometric.__version__)"
python -c "import networkx; print('NetworkX:', networkx.__version__)"
python -c "import loguru; print('Loguru installed successfully')"
```

### PyTorch Geometric Installation Issues

**Problem**: `PyTorch Geometric not installed`

**Alternative Installation Methods**:

1. **Method 1 - Conda (Recommended)**:
   ```bash
   conda install pyg -c pyg
   ```

2. **Method 2 - Pip with wheel**:
   ```bash
   pip install torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
   ```

3. **Method 3 - From source**:
   ```bash
   pip install git+https://github.com/pyg-team/pytorch_geometric.git
   ```

### CUDA Configuration

**Issue**: `CUDA not available - using CPU`

**Solutions**:

1. **CPU Mode (Current)**:
   - The system works with CPU
   - Slower but functional
   - No action needed

2. **Enable CUDA (Optional)**:
   ```bash
   # Check CUDA availability
   nvidia-smi
   
   # Install CUDA-enabled PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Configuration Issues

**Problem**: `Config not available`

**Solution**:

1. **Verify config.yaml exists**:
   ```bash
   ls -la config.yaml
   ```

2. **Check module imports**:
   ```bash
   python -c "from src.utils import Config; print('Config module working')"
   ```

## Complete Setup Verification

After fixing issues, run complete verification:

```bash
# 1. Check Python version
python --version

# 2. Run diagnostic
python diagnostic.py

# 3. Test imports
python -c "
import torch
import torch_geometric
import networkx
import loguru
print('All major dependencies available')
"

# 4. Test system
python main.py --help
```

## Data Verification

Ensure your data files are correctly placed:

```bash
# Check data structure
ls -la data/raw/
# Should show:
# train_transaction.csv (651.7 MB)
# train_identity.csv (25.3 MB)  
# test_transaction.csv (584.8 MB)
# test_identity.csv (24.6 MB)

# Verify file integrity
python -c "
import pandas as pd
df = pd.read_csv('data/raw/train_transaction.csv', nrows=5)
print(f'Transaction data shape preview: {df.shape}')
print('Data loaded successfully')
"
```

## Quick Fix Script

Run this comprehensive fix script:

```bash
#!/bin/bash
# quick_fix.sh

echo "Starting GNN Fraud Detection setup..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
echo "Python version: $python_version"

# Install/upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric networkx loguru
pip install -r requirements.txt

# Verify installation
python3 diagnostic.py

echo "Setup complete! Check diagnostic results above."
```

## Getting Help

If issues persist:

1. **Check diagnostic output**: `python diagnostic.py`
2. **Verify requirements**: Compare with `requirements.txt`
3. **Environment issues**: Try fresh virtual environment
4. **Data issues**: Re-download from Kaggle
5. **PyTorch Geometric**: Try different installation methods above

## Success Indicators

System is ready when diagnostic shows:
- ✅ Python 3.8+
- ✅ All dependencies installed
- ✅ Data files present and correct size
- ✅ PyTorch and PyTorch Geometric working
- ✅ All modules importable

Expected final message: `SYSTEM FULLY FUNCTIONAL!`
