#!/usr/bin/env python3
"""
Initialization and test script for the fraud detection system.

This script runs a complete system demonstration,
including data loading, training and evaluation.
"""

import sys
from pathlib import Path
import subprocess

# Add root directory to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))


def check_dependencies():
    """Check if all dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        ('torch', 'torch'),
        ('torch_geometric', 'torch_geometric'), 
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('networkx', 'networkx'),
        ('yaml', 'pyyaml'),
        ('loguru', 'loguru')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} not found")
    
    # Additional specific checks
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 PyTorch device: {device}")
    except Exception as e:
        print(f"⚠️ Aviso PyTorch: {e}")
    
    if missing_packages:
        print(f"Pacotes faltando: {missing_packages}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    print("✅ Todas as dependências estão instaladas")
    return True


def check_data():
    """Verifica se os dados estão disponíveis."""
    print("Verificando disponibilidade dos dados...")
    
    data_dir = Path("ieee-fraud-detection")
    required_files = [
        "train_transaction.csv",
        "train_identity.csv", 
        "test_transaction.csv",
        "test_identity.csv"
    ]
    
    if not data_dir.exists():
        print(f"❌ Diretório de dados não encontrado: {data_dir}")
        print("Baixe o dataset IEEE-CIS Fraud Detection do Kaggle")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✅ {file} encontrado")
    
    if missing_files:
        print(f"❌ Arquivos faltando: {missing_files}")
        return False
    
    # Verificar arquivo de configuração
    config_file = Path("config.yaml")
    if config_file.exists():
        print("✅ config.yaml encontrado")
    else:
        print("⚠️ config.yaml não encontrado - será criado automaticamente")
    
    print("✅ Todos os arquivos de dados estão disponíveis")
    return True


def run_demo():
    """Executa demonstração completa do sistema."""
    print("🚀 Iniciando demonstração do sistema...")
    
    try:
        # Importar aqui para evitar problemas de dependência
        print("Importando módulos do sistema...")
        from main import FraudDetectionSystem
        
        # Inicializar sistema
        print("Inicializando sistema...")
        system = FraudDetectionSystem("config.yaml")
        
        # Configurar para usar amostra pequena (demo)
        system.config.set('data.sample_size', 1000)
        system.config.set('training.epochs', 5)
        
        print("Configuração de demo aplicada:")
        print("- Sample size: 1000 transações")
        print("- Epochs: 5 (para demonstração rápida)")
        
        # Executar pipeline completo
        print("Executando pipeline de demonstração...")
        results = system.run_full_pipeline()
        
        print("✅ Demonstração concluída com sucesso!")
        print("📊 Resumo dos resultados:")
        if isinstance(results, dict):
            for key, value in results.items():
                if key != 'error':
                    print(f"   - {key}: {value}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {str(e)}")
        print("Verifique se todas as dependências estão instaladas")
        return False
    except FileNotFoundError as e:
        print(f"❌ Arquivo não encontrado: {str(e)}")
        print("Verifique se os dados estão na pasta ieee-fraud-detection/")
        return False
    except Exception as e:
        print(f"❌ Erro na demonstração: {str(e)}")
        print(f"Tipo de erro: {type(e).__name__}")
        return False


def launch_dashboard():
    """Lança o dashboard Streamlit."""
    print("🌐 Lançando dashboard...")
    
    try:
        dashboard_path = Path("dashboard/app.py")
        venv_python = Path(".venv/bin/python")
        
        if not dashboard_path.exists():
            print(f"❌ Dashboard não encontrado: {dashboard_path}")
            return False
        
        # Verificar se streamlit está disponível
        try:
            import streamlit
            print(f"✅ Streamlit {streamlit.__version__} disponível")
        except ImportError:
            print("❌ Streamlit não está instalado")
            return False
        
        # Usar ambiente virtual se disponível
        if venv_python.exists():
            cmd = [str(venv_python), "-m", "streamlit", "run", str(dashboard_path), "--server.port", "8501"]
            print("🐍 Usando ambiente virtual")
        else:
            cmd = ["streamlit", "run", str(dashboard_path), "--server.port", "8501"]
            print("🌐 Usando ambiente global")
        
        print("Executando comando: " + " ".join(cmd))
        print("Dashboard será aberto em: http://localhost:8501")
        print("Pressione Ctrl+C para parar o dashboard")
        print("-" * 50)
        
        # Executar em processo separado
        subprocess.run(cmd)
        
        return True
        
    except KeyboardInterrupt:
        print("\n✅ Dashboard interrompido pelo usuário")
        return True
    except FileNotFoundError as e:
        print(f"❌ Comando não encontrado: {e}")
        print("Verifique se o Streamlit está instalado")
        return False
    except Exception as e:
        print(f"❌ Erro ao lançar dashboard: {str(e)}")
        return False


def print_usage():
    """Imprime instruções de uso."""
    print("""
🔍 Sistema de Detecção de Fraude com Graph Neural Networks

Uso: python setup.py [opção]

Opções:
  check     - Verificar dependências e dados
  demo      - Executar demonstração rápida do sistema
  dashboard - Lançar dashboard interativo
  full      - Executar pipeline completo de produção
  help      - Mostrar esta mensagem

Exemplos:
  python setup.py check       # Verificar sistema
  python setup.py demo        # Demo rápida (5 epochs, 1000 amostras)
  python setup.py dashboard   # Lançar interface web
  python setup.py full        # Treinamento completo

Para mais informações, consulte README.md
""")


def main():
    """Função principal."""
    print("🔍 Sistema de Detecção de Fraude - Setup e Demo")
    print("=" * 50)
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help":
        print_usage()
        return
    
    elif command == "check":
        print("Executando verificações do sistema...")
        
        deps_ok = check_dependencies()
        data_ok = check_data()
        
        if deps_ok and data_ok:
            print("✅ Sistema pronto para uso!")
        else:
            print("❌ Sistema não está pronto. Corrija os problemas acima.")
    
    elif command == "demo":
        print("Executando demonstração rápida...")
        
        # Verificar pré-requisitos
        if not check_dependencies():
            print("Dependências não satisfeitas")
            return
        
        if not check_data():
            print("Dados não disponíveis")
            return
        
        # Executar demo
        success = run_demo()
        
        if success:
            print("🎉 Demonstração concluída!")
            print("Execute 'python setup.py dashboard' para ver a interface")
        else:
            print("Demonstração falhou")
    
    elif command == "dashboard":
        print("Lançando dashboard...")
        launch_dashboard()
    
    elif command == "full":
        print("Executando pipeline completo...")
        
        # Verificar pré-requisitos
        if not check_dependencies():
            return
        
        if not check_data():
            return
        
        try:
            # Importar aqui para evitar problemas de dependência
            from main import FraudDetectionSystem
            
            # Executar sistema completo
            system = FraudDetectionSystem("config.yaml")
            system.run_full_pipeline()
            
            print("✅ Pipeline completo concluído!")
            print("Resultados salvos em: results/")
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {str(e)}")
    
    else:
        print(f"Comando desconhecido: {command}")
        print_usage()


if __name__ == "__main__":
    main()
