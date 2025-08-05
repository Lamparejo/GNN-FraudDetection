#!/usr/bin/env python3
"""
Script de teste final do sistema de detecção de fraude.

Este script executa todos os testes necessários para garantir
que o sistema está funcionando corretamente.
"""

import sys
import subprocess

def run_command(cmd, description):
    """Executa comando e reporta resultado."""
    print(f"\n🔧 {description}")
    print(f"Comando: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} - Sucesso")
            if result.stdout:
                print(result.stdout[-500:])  # Últimas 500 chars
        else:
            print(f"❌ {description} - Falhou")
            print(f"Erro: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - Timeout")
        return False
    except Exception as e:
        print(f"❌ {description} - Erro: {e}")
        return False
    
    return True

def main():
    """Função principal de teste."""
    print("🧪 TESTE FINAL DO SISTEMA")
    print("=" * 60)
    
    # Lista de testes
    tests = [
        (["python", "setup.py", "check"], "Verificação do Sistema"),
        (["python", "diagnostic.py"], "Diagnóstico Completo"),
        (["python", "-c", "import torch, torch_geometric, pandas, streamlit; print('Imports OK')"], "Teste de Imports"),
        (["python", "-c", "from main import FraudDetectionSystem; print('Sistema OK')"], "Teste do Sistema Principal"),
    ]
    
    # Executar testes
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📋 RELATÓRIO FINAL")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{status:12} {desc}")
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("\nPróximos passos:")
        print("1. python setup.py demo           # Executar demonstração")
        print("2. python setup.py dashboard      # Lançar interface web")
        print("3. python setup.py full           # Pipeline completo")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("Verifique os erros acima antes de continuar")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
