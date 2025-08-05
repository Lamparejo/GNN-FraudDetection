#!/usr/bin/env python3
"""
Complete system diagnostic script.

Verifies all aspects of the fraud detection system
including dependencies, data, configuration and functionality.
"""

import sys
from pathlib import Path
import importlib.util

# Add root directory to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))


class SystemDiagnostic:
    """Complete system diagnostic class."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def log_error(self, message: str):
        """Log error message."""
        self.errors.append(message)
        print(f"ERROR: {message}")
        
    def log_warning(self, message: str):
        """Log warning message."""
        self.warnings.append(message)
        print(f"WARNING: {message}")
        
    def log_info(self, message: str):
        """Log information message."""
        self.info.append(message)
        print(f"INFO: {message}")
        
    def check_python_environment(self) -> bool:
        """Check Python environment."""
        print("\nChecking Python environment...")
        
        # Python version
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log_info(f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.log_error(f"Python {version.major}.{version.minor} not supported (requires 3.8+)")
            return False
        
        # Pip
        try:
            import pip
            self.log_info(f"pip {pip.__version__}")
        except ImportError:
            self.log_warning("pip not found")
        
        # Virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log_info("Running in virtual environment")
        else:
            self.log_warning("Not running in virtual environment")
        
        return True
        
    def check_dependencies_detailed(self) -> bool:
        """Check dependencies with details."""
        print("\nChecking dependencies...")
        
        # Package list with minimum versions
        packages = [
            ('torch', '2.0.0', 'PyTorch'),
            ('torch_geometric', '2.4.0', 'PyTorch Geometric'),
            ('pandas', '1.5.0', 'Pandas'),
            ('numpy', '1.21.0', 'NumPy'),
            ('sklearn', '1.0.0', 'Scikit-learn'),
            ('streamlit', '1.28.0', 'Streamlit'),
            ('plotly', '5.15.0', 'Plotly'),
            ('networkx', '3.0', 'NetworkX'),
            ('yaml', '6.0', 'PyYAML'),
            ('loguru', '0.7.0', 'Loguru')
        ]
        
        missing = []
        for import_name, min_version, display_name in packages:
            try:
                module = __import__(import_name)
                if hasattr(module, '__version__'):
                    version = module.__version__
                    self.log_info(f"{display_name} {version}")
                else:
                    self.log_info(f"{display_name} (version not detected)")
            except ImportError:
                missing.append(display_name)
                self.log_error(f"{display_name} not installed")
        
        if missing:
            package_names = [pkg.lower() for pkg in missing]
            self.log_error(f"Install packages: pip install {' '.join(package_names)}")
            return False
        
        return True
        
    def check_pytorch_specific(self) -> bool:
        """Check PyTorch specific configuration."""
        print("\nChecking PyTorch...")
        
        try:
            import torch
            
            # Version
            self.log_info(f"PyTorch {torch.__version__}")
            
            # CUDA
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_info(f"CUDA available: {device_count} GPU(s)")
                self.log_info(f"Primary GPU: {device_name} ({memory:.1f} GB)")
            else:
                self.log_warning("CUDA not available - using CPU")
            
            # Basic test
            x = torch.randn(2, 3)
            _ = x + 1  # Basic operation
            self.log_info("PyTorch basic operations working")
            
            # PyTorch Geometric
            try:
                import torch_geometric
                self.log_info(f"PyTorch Geometric {torch_geometric.__version__}")
                
                # Basic PyG test
                from torch_geometric.data import Data
                edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
                x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
                _ = Data(x=x, edge_index=edge_index)  # Create Data object
                self.log_info("PyTorch Geometric working correctly")
                
            except ImportError as e:
                self.log_error(f"PyTorch Geometric not installed: {e}")
                return False
                
        except Exception as e:
            self.log_error(f"PyTorch error: {e}")
            return False
        
        return True
        
    def check_data_files(self) -> bool:
        """Check data files."""
        print("\nChecking data...")
        
        data_dir = Path("data/raw")
        if not data_dir.exists():
            self.log_error(f"Directory {data_dir} not found")
            self.log_info("Download dataset from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection")
            return False
            
        required_files = [
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv", 
            "test_identity.csv"
        ]
        
        for file in required_files:
            file_path = data_dir / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / 1024 / 1024
                self.log_info(f"{file} ({size_mb:.1f} MB)")
            else:
                self.log_error(f"{file} not found")
                return False
        
        return True
        
    def check_project_structure(self) -> bool:
        """Check project structure."""
        print("\nChecking project structure...")
        
        required_dirs = [
            "src",
            "src/data",
            "src/models", 
            "src/training",
            "src/utils",
            "src/visualization",
            "webapp",
            "docs"
        ]
        
        required_files = [
            "main.py",
            "config.yaml",
            "requirements.txt",
            "README.md",
            "webapp/app.py"
        ]
        
        # Check directories
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                self.log_info(f"Directory {dir_path}")
            else:
                self.log_warning(f"Directory {dir_path} not found")
        
        # Check files
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_info(f"File {file_path}")
            else:
                self.log_warning(f"File {file_path} not found")
        
        return True
        
    def check_functionality(self) -> bool:
        """Test basic functionality."""
        print("\nTesting functionality...")
        
        try:
            # Test importing main modules
            try:
                import importlib.util
                spec = importlib.util.find_spec("src.utils")
                if spec is not None:
                    from src.utils import Config
                    self.log_info("Config imported")
                    
                    # Test configuration
                    _ = Config("config.yaml")
                    self.log_info("Configuration loaded")
                else:
                    self.log_warning("src.utils module not found")
            except Exception:
                self.log_warning("Config not available")
            
            try:
                import importlib.util
                spec = importlib.util.find_spec("src.data")
                if spec is not None:
                    self.log_info("Data module available")
                else:
                    self.log_warning("Data module not available")
            except Exception:
                self.log_warning("DataLoader not available")
            
            try:
                import importlib.util
                spec = importlib.util.find_spec("src.models")
                if spec is not None:
                    self.log_info("Models module available")
                else:
                    self.log_warning("Models module not available")
            except Exception:
                self.log_warning("BaseGNNModel not available")
            
            return True
            
        except Exception as e:
            self.log_error(f"Functionality error: {e}")
            return False
            
    def generate_report(self):
        """Generate final report."""
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)
        
        print(f"\nINFO: {len(self.info)}")
        print(f"WARNINGS: {len(self.warnings)}")  
        print(f"ERRORS: {len(self.errors)}")
        
        if self.errors:
            print("\nERRORS FOUND:")
            for error in self.errors:
                print(f"   • {error}")
                
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # General status
        if not self.errors:
            if not self.warnings:
                print("\nSYSTEM FULLY FUNCTIONAL!")
                print("Run: python main.py")
            else:
                print("\nSYSTEM FUNCTIONAL WITH WARNINGS")
                print("Run: python main.py")
        else:
            print("\nSYSTEM HAS PROBLEMS")
            print("Fix errors before continuing")
            
    def run_full_diagnostic(self):
        """Run complete diagnostic."""
        print("COMPLETE SYSTEM DIAGNOSTIC")
        print("="*50)
        
        checks = [
            ("Python Environment", self.check_python_environment),
            ("Dependencies", self.check_dependencies_detailed),
            ("PyTorch", self.check_pytorch_specific),
            ("Data Files", self.check_data_files),
            ("Project Structure", self.check_project_structure),
            ("Functionality", self.check_functionality)
        ]
        
        results = {}
        for name, check_func in checks:
            try:
                results[name] = check_func()
            except Exception as e:
                self.log_error(f"Error in {name} check: {e}")
                results[name] = False
        
        self.generate_report()
        
        return all(results.values())


def main():
    """Main function."""
    diagnostic = SystemDiagnostic()
    success = diagnostic.run_full_diagnostic()
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
