#!/usr/bin/env python3
"""
Automatic installation script for the fraud detection system.

This script automatically sets up the virtual environment and 
installs all necessary dependencies.
"""

import sys
import os
import subprocess
import venv
from pathlib import Path


def create_virtual_environment():
    """Create virtual environment."""
    print("🐍 Creating virtual environment...")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created in .venv/")
        return True
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False


def get_python_executable():
    """Get Python executable path in virtual environment."""
    if os.name == 'nt':  # Windows
        return Path(".venv/Scripts/python.exe")
    else:  # Linux/Mac
        return Path(".venv/bin/python")


def install_dependencies():
    """Install dependencies in virtual environment."""
    print("📦 Installing dependencies...")
    
    python_exe = get_python_executable()
    if not python_exe.exists():
        print("❌ Python executable not found in virtual environment")
        return False
    
    try:
        # Atualizar pip
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("✅ pip atualizado")
        
        # Instalar torch primeiro (pode ser necessário para PyG)
        subprocess.run([str(python_exe), "-m", "pip", "install", "torch", "torchvision"], 
                      check=True, capture_output=True)
        print("✅ PyTorch instalado")
        
        # Instalar PyTorch Geometric
        subprocess.run([str(python_exe), "-m", "pip", "install", "torch-geometric"], 
                      check=True, capture_output=True)
        print("✅ PyTorch Geometric instalado")
        
        # Instalar demais dependências
        if Path("requirements.txt").exists():
            subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True, capture_output=True)
            print("✅ Dependências do requirements.txt instaladas")
        else:
            # Lista de pacotes essenciais
            packages = [
                "pandas>=1.5.0",
                "numpy>=1.21.0", 
                "scikit-learn>=1.0.0",
                "streamlit>=1.28.0",
                "plotly>=5.15.0",
                "networkx>=3.0",
                "pyyaml>=6.0",
                "loguru>=0.7.0"
            ]
            
            for package in packages:
                subprocess.run([str(python_exe), "-m", "pip", "install", package], 
                              check=True, capture_output=True)
                print(f"✅ {package} instalado")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na instalação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def verify_installation():
    """Verifica se a instalação foi bem-sucedida."""
    print("🔍 Verificando instalação...")
    
    python_exe = get_python_executable()
    
    test_imports = [
        "import torch; print(f'PyTorch {torch.__version__}')",
        "import torch_geometric; print(f'PyG {torch_geometric.__version__}')",
        "import pandas; print(f'Pandas {pandas.__version__}')",
        "import streamlit; print(f'Streamlit {streamlit.__version__}')",
        "import plotly; print(f'Plotly {plotly.__version__}')"
    ]
    
    success = True
    for test in test_imports:
        try:
            result = subprocess.run([str(python_exe), "-c", test], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            package_name = test.split()[1].replace(';', '')
            print(f"❌ Falha ao importar {package_name}")
            success = False
    
    return success


def create_activation_script():
    """Cria script de ativação do ambiente."""
    if os.name == 'nt':  # Windows
        script_content = """@echo off
echo Ativando ambiente virtual...
call .venv\\Scripts\\activate.bat
echo ✅ Ambiente ativado!
echo.
echo Comandos disponíveis:
echo   python setup.py check     - Verificar sistema
echo   python setup.py demo      - Executar demo
echo   python setup.py dashboard - Lançar dashboard
echo.
cmd /k
"""
        script_path = "activate.bat"
    else:  # Linux/Mac
        script_content = """#!/bin/bash
echo "Ativando ambiente virtual..."
source .venv/bin/activate
echo "✅ Ambiente ativado!"
echo ""
echo "Comandos disponíveis:"
echo "  python setup.py check     - Verificar sistema"
echo "  python setup.py demo      - Executar demo"
echo "  python setup.py dashboard - Lançar dashboard"
echo ""
exec bash
"""
        script_path = "activate.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✅ Script de ativação criado: {script_path}")


def main():
    """Função principal de instalação."""
    print("🚀 INSTALAÇÃO AUTOMÁTICA - Sistema de Detecção de Fraude")
    print("=" * 60)
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Passos da instalação
    steps = [
        ("Criar ambiente virtual", create_virtual_environment),
        ("Instalar dependências", install_dependencies), 
        ("Verificar instalação", verify_installation),
        ("Criar script de ativação", create_activation_script)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"❌ Falha em: {step_name}")
            sys.exit(1)
    
    # Sucesso
    print("\n" + "=" * 60)
    print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    
    print("\n📝 Próximos passos:")
    if os.name == 'nt':
        print("1. Execute: activate.bat")
    else:
        print("1. Execute: ./activate.sh")
        print("   ou: source .venv/bin/activate")
    
    print("2. Baixe os dados do Kaggle para ieee-fraud-detection/")
    print("3. Execute: python setup.py check")
    print("4. Execute: python setup.py demo")
    
    print("\n📚 Consulte QUICK_START.md para mais informações")


if __name__ == "__main__":
    main()
