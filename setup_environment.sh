#!/bin/bash
# setup_environment.sh - Automated environment setup for GNN Fraud Detection

set -e  # Exit on any error

echo "GNN Fraud Detection - Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 8 ]; then
    print_success "Python $python_version detected (compatible)"
else
    print_error "Python $python_version detected. Python 3.8+ required!"
    print_warning "Please upgrade Python and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default)
print_status "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
print_status "Installing PyTorch Geometric..."
pip install torch-geometric

# Install other requirements
print_status "Installing project requirements..."
pip install -r requirements.txt

# Verify critical installations
print_status "Verifying installations..."

# Test PyTorch
python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null && print_success "PyTorch working" || print_error "PyTorch installation failed"

# Test PyTorch Geometric
python3 -c "import torch_geometric; print('PyTorch Geometric version:', torch_geometric.__version__)" 2>/dev/null && print_success "PyTorch Geometric working" || print_error "PyTorch Geometric installation failed"

# Test NetworkX
python3 -c "import networkx; print('NetworkX version:', networkx.__version__)" 2>/dev/null && print_success "NetworkX working" || print_error "NetworkX installation failed"

# Test Loguru
python3 -c "import loguru; print('Loguru available')" 2>/dev/null && print_success "Loguru working" || print_error "Loguru installation failed"

# Create data directories if they don't exist
print_status "Setting up data directories..."
mkdir -p data/raw data/processed data/graphs
mkdir -p results/models/checkpoints results/models/trained
mkdir -p logs/training logs/inference

# Create .gitkeep files
touch data/raw/.gitkeep data/processed/.gitkeep data/graphs/.gitkeep
touch results/models/.gitkeep results/models/checkpoints/.gitkeep results/models/trained/.gitkeep

print_success "Data directories created"

# Run diagnostic
print_status "Running system diagnostic..."
python3 diagnostic.py

echo ""
echo "=========================================="
print_success "Environment setup complete!"
echo ""
print_warning "Next steps:"
echo "1. Download IEEE-CIS Fraud Detection dataset from Kaggle"
echo "2. Place CSV files in data/raw/ directory"
echo "3. Run 'python diagnostic.py' to verify complete setup"
echo "4. Run 'python main.py' to start the system"
echo ""
print_status "To activate environment in future sessions:"
echo "source venv/bin/activate"
