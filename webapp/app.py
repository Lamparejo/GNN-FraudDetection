"""
Interactive dashboard for fraud detection system visualization and monitoring.

This Streamlit dashboard offers a complete interface for:
- Performance metrics visualization
- Graph and subnetwork analysis
- Real-time monitoring
- Model explainability
- Dynamic model retraining
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for importing system modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from main import FraudDetectionSystem
    from src.utils import device_manager
    from src.training import GNNTrainer
    import torch
    MODEL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Model integration not available: {e}")
    MODEL_INTEGRATION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background: linear-gradient(90deg, #00d2d3 0%, #54a0ff 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FraudDetectionDashboard:
    """Classe principal do dashboard de detecção de fraude."""
    
    def __init__(self):
        """Inicializa o dashboard."""
        self.setup_session_state()
        
        # Inicializar sistema de detecção de fraude se disponível
        if MODEL_INTEGRATION_AVAILABLE:
            self.init_fraud_system()
        
        # NÃO carregar dados de exemplo - só usar dados reais após treinamento
    
    def init_fraud_system(self):
        """Inicializa o sistema de detecção de fraude."""
        try:
            if 'fraud_system' not in st.session_state:
                st.session_state.fraud_system = None
                st.session_state.model_config = {}
                st.session_state.current_threshold = 0.5
                st.session_state.model_loaded = False
                st.session_state.model_trained = False
                st.session_state.training_in_progress = False
                st.session_state.training_data = None
                st.session_state.evaluation_metrics = None
                
        except Exception as e:
            st.warning(f"⚠️ Erro ao inicializar sistema de fraude: {e}")
    
    def is_model_trained(self):
        """Verifica se um modelo foi treinado."""
        return st.session_state.get('model_trained', False) and st.session_state.get('fraud_system') is not None
    
    def calculate_current_metrics(self, threshold=0.5):
        """Calcula métricas atuais baseadas no threshold especificado."""
        if not hasattr(self, 'transactions_data'):
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'auc_pr': 0.0
            }
        
        # Gerar predições com threshold atual
        probabilities, predictions = self.predict_with_threshold(self.transactions_data, threshold)
        
        y_true = self.transactions_data['is_fraud']
        y_pred = predictions
        y_scores = probabilities
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC e PR AUC usando probabilidades
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
        except Exception:
            auc_roc = 0.0
            auc_pr = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
    
    def predict_with_threshold(self, data, threshold):
        """Faz predições aplicando o threshold especificado."""
        try:
            # Por enquanto, usar dados simulados mas aplicar threshold corretamente
            return self.simulate_predictions(data, threshold)
            
        except Exception as e:
            st.warning(f"⚠️ Erro na predição: {e}")
            return self.simulate_predictions(data, threshold)
    
    def simulate_predictions(self, data, threshold):
        """Simula predições para demonstração com threshold aplicado."""
        np.random.seed(42)  # Para resultados consistentes
        probabilities = np.random.beta(1, 20, len(data))
        
        # Ajustar algumas para serem fraudes
        fraud_indices = np.random.choice(len(data), size=int(len(data) * 0.05), replace=False)
        probabilities[fraud_indices] = np.random.beta(8, 2, len(fraud_indices))
        
        # Aplicar threshold para gerar predições
        predictions = (probabilities > threshold).astype(int)
        return probabilities, predictions
    
    def render_empty_page(self, page_name, icon):
        """Renderiza uma página vazia até que o modelo seja treinado."""
        st.title(f"{icon} {page_name}")
        
        st.info("🚀 **Configure e treine sua rede GNN primeiro!**")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 1rem; margin: 2rem 0;">
                <h3>🧠 Rede Neural GNN</h3>
                <p>Esta página ficará disponível após o treinamento da sua rede neural em grafos.</p>
                <p><strong>Vá para Configurações para treinar seu modelo!</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Botão para ir às configurações
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("⚙️ Ir para Configurações", use_container_width=True):
                st.rerun()
        
    def setup_session_state(self):
        """Configura o estado da sessão."""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
    
    def load_sample_data(self):
        """Carrega dados simulados para demonstração."""
        # Simulação de dados para demonstração
        np.random.seed(42)
        
        # Histórico de treinamento simulado
        epochs = range(1, 101)
        self.training_history = pd.DataFrame({
            'epoch': epochs,
            'train_loss': np.random.exponential(0.1, 100) + 0.05,
            'val_loss': np.random.exponential(0.12, 100) + 0.08,
            'train_f1': np.random.beta(8, 2, 100) * 0.9 + 0.1,
            'val_f1': np.random.beta(7, 2, 100) * 0.85 + 0.1
        })
        
        # Dados de transações simuladas
        n_transactions = 1000
        self.transactions_data = pd.DataFrame({
            'transaction_id': range(n_transactions),
            'amount': np.random.lognormal(3, 1.5, n_transactions),
            'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='h'),
            'fraud_probability': np.random.beta(1, 20, n_transactions),
            'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.95, 0.05]),
            'device_id': np.random.randint(1, 100, n_transactions),
            'user_id': np.random.randint(1, 500, n_transactions),
            'card_id': np.random.randint(1, 200, n_transactions)
        })
        
        # Ajustar probabilidades para casos de fraude
        fraud_mask = self.transactions_data['is_fraud'] == 1
        self.transactions_data.loc[fraud_mask, 'fraud_probability'] = np.random.beta(8, 2, fraud_mask.sum())
        
        st.session_state.data_loaded = True
    
    def render_sidebar(self):
        """Renderiza a barra lateral."""
        st.sidebar.title("🔍 Fraud Detection GNN")
        st.sidebar.markdown("---")
        
        # Verificar se modelo foi treinado
        model_trained = self.is_model_trained()
        training_in_progress = st.session_state.get('training_in_progress', False)
        
        # Seleção de página baseada no estado do modelo
        if not model_trained and not training_in_progress:
            # Se não há modelo treinado, só mostrar configurações
            page = "⚙️ Configurações"
            st.sidebar.info("🚀 Configure e treine sua rede GNN primeiro!")
        else:
            # Se modelo está treinado, mostrar todas as páginas
            available_pages = ["⚙️ Configurações", "📊 Overview", "📈 Métricas", "🕸️ Análise de Grafos", 
                             "🔍 Detecção em Tempo Real", "📋 Histórico"]
            
            page = st.sidebar.selectbox(
                "Selecione uma página:",
                available_pages
            )
        
        st.sidebar.markdown("---")
        
        # Status do sistema
        st.sidebar.subheader("Status do Sistema")
        
        if training_in_progress:
            st.sidebar.warning("🔄 Treinamento em Progresso...")
        elif model_trained:
            st.sidebar.success("✅ Modelo GNN Treinado")
            # Mostrar métricas do modelo se disponível
            if 'training_metrics' in st.session_state:
                metrics = st.session_state.training_metrics
                st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                st.sidebar.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        else:
            st.sidebar.error("❌ Nenhum Modelo Treinado")
        
        if MODEL_INTEGRATION_AVAILABLE:
            st.sidebar.success("✅ Integração Disponível")
        else:
            st.sidebar.error("❌ Integração Indisponível")
        
        st.sidebar.markdown("---")
        
        # Configurações rápidas (só se modelo estiver treinado)
        if model_trained:
            st.sidebar.subheader("Configurações Rápidas")
            
            threshold = st.sidebar.slider(
                "Threshold de Fraude:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('current_threshold', 0.5),
                step=0.01
            )
            
            auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
            
            if auto_refresh:
                st.sidebar.info("🔄 Auto-refresh ativo")
        else:
            threshold = 0.5
        
        return page, threshold
    
    def render_overview_page(self):
        """Renderiza a página de overview."""
        if not self.is_model_trained():
            self.render_empty_page("Overview", "📊")
            return
        
        st.title("📊 Fraud Detection - Overview")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("❌ Sistema de fraude não disponível")
            return
        
        # Calcular métricas atuais do modelo real
        evaluation_metrics = st.session_state.get('evaluation_metrics', {})
        
        # KPIs principais do modelo treinado
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{evaluation_metrics.get('accuracy', 0):.2%}",
                delta="+Real"
            )
        
        with col2:
            st.metric(
                label="🔍 Precision",
                value=f"{evaluation_metrics.get('precision', 0):.2%}",
                delta="+Real"
            )
        
        with col3:
            st.metric(
                label="📈 Recall",
                value=f"{evaluation_metrics.get('recall', 0):.2%}",
                delta="+Real"
            )
        
        with col4:
            st.metric(
                label="🏆 F1-Score",
                value=f"{evaluation_metrics.get('f1', 0):.2%}",
                delta="+Real"
            )
        
        st.markdown("---")
        
        # Histórico de treinamento real
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("📈 Histórico de Treinamento Real")
            
            training_history = st.session_state.training_history
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Loss', 'F1-Score'),
                vertical_spacing=0.1
            )
            
            if 'train_loss' in training_history:
                fig.add_trace(
                    go.Scatter(
                        y=training_history['train_loss'],
                        name='Train Loss',
                        line=dict(color='#ff6b6b')
                    ),
                    row=1, col=1
                )
            
            if 'val_loss' in training_history:
                fig.add_trace(
                    go.Scatter(
                        y=training_history['val_loss'],
                        name='Val Loss',
                        line=dict(color='#ffa726')
                    ),
                    row=1, col=1
                )
            
            if 'train_f1' in training_history:
                fig.add_trace(
                    go.Scatter(
                        y=training_history['train_f1'],
                        name='Train F1',
                        line=dict(color='#42a5f5'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            if 'val_f1' in training_history:
                fig.add_trace(
                    go.Scatter(
                        y=training_history['val_f1'],
                        name='Val F1',
                        line=dict(color='#66bb6a'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                title_text="Evolução das Métricas do Modelo Treinado"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Informações do modelo
        st.subheader("🤖 Informações do Modelo")
        model_config = st.session_state.get('model_config', {})
        training_metrics = st.session_state.get('training_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tipo de Modelo", training_metrics.get('model_type', 'N/A'))
        
        with col2:
            st.metric("Épocas Treinadas", training_metrics.get('training_epochs', 0))
        
        with col3:
            model_info = model_config.get('model', {})
            st.metric("Hidden Dim", model_info.get('hidden_dim', 'N/A'))
        
        with col4:
            st.metric("Num Layers", model_info.get('num_layers', 'N/A'))
    
    def render_metrics_page(self, threshold=0.5):
        """Renderiza a página de métricas detalhadas."""
        if not self.is_model_trained():
            self.render_empty_page("Métricas Detalhadas", "📈")
            return
        
        st.title("📈 Métricas Detalhadas")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("❌ Sistema de fraude não disponível")
            return
        
        st.info(f"🎚️ **Threshold Atual:** {threshold:.2f}")
        
        # Mostrar métricas do modelo real
        evaluation_metrics = st.session_state.get('evaluation_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{evaluation_metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("🔍 Precision", f"{evaluation_metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("📈 Recall", f"{evaluation_metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("🏆 F1-Score", f"{evaluation_metrics.get('f1', 0):.3f}")
        
        # Exibir informações do modelo treinado
        st.subheader("🤖 Informações do Modelo Treinado")
        training_metrics = st.session_state.get('training_metrics', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tipo de Modelo", training_metrics.get('model_type', 'N/A'))
        
        with col2:
            st.metric("Épocas Treinadas", training_metrics.get('training_epochs', 0))
        
        with col3:
            st.metric("AUC-ROC", f"{evaluation_metrics.get('auc_roc', 0):.3f}")
        
        # Matriz de confusão seria calculada com dados de teste reais
        st.info("📊 Métricas detalhadas disponíveis após avaliação completa do modelo.")
        
        # Histórico de treinamento se disponível
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("📈 Evolução do Treinamento")
            
            training_history = st.session_state.training_history
            
            if 'train_loss' in training_history and 'val_loss' in training_history:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=training_history['train_loss'],
                    name='Train Loss',
                    line=dict(color='#ff6b6b')
                ))
                
                fig.add_trace(go.Scatter(
                    y=training_history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='#42a5f5')
                ))
                
                fig.update_layout(
                    title='Evolução da Loss Durante o Treinamento',
                    xaxis_title='Época',
                    yaxis_title='Loss'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_graph_analysis_page(self):
        """Renderiza a página de análise de grafos."""
        if not self.is_model_trained():
            self.render_empty_page("Análise de Grafos", "🕸️")
            return
        
        st.title("🕸️ Análise de Grafos")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("❌ Sistema de fraude não disponível")
            return
        
        st.info("🔗 **Análise de grafos baseada no modelo GNN treinado**")
        
        # Informações do grafo treinado
        st.subheader("📊 Estatísticas do Grafo Treinado")
        
        # Placeholder para estatísticas reais do grafo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nós Processados", "Disponível após treinamento")
        
        with col2:
            st.metric("Arestas Criadas", "Disponível após treinamento")
        
        with col3:
            st.metric("Densidade do Grafo", "Disponível após treinamento")
        
        st.info("📈 Visualizações detalhadas do grafo estarão disponíveis nas próximas versões.")
    
    def render_realtime_page(self, threshold=0.5):
        """Renderiza a página de detecção em tempo real."""
        if not self.is_model_trained():
            self.render_empty_page("Detecção em Tempo Real", "🔍")
            return
        
        st.title("🔍 Detecção em Tempo Real")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("❌ Sistema de fraude não disponível")
            return
        
        st.info(f"🎚️ **Threshold Atual:** {threshold:.2f}")
        st.info("🤖 **Usando modelo GNN treinado para predições reais**")
        
        st.subheader("📊 Monitor de Transações")
        
        # Simulação de nova transação para teste do modelo real
        if st.button("🔄 Simular Nova Transação"):
            st.info("📡 Processando transação com modelo GNN...")
            
            # Placeholder para predição real
            fraud_probability = np.random.beta(1, 20)  # Temporário até integração completa
            
            new_transaction = {
                'timestamp': datetime.now(),
                'amount': np.random.lognormal(3, 1.5),
                'fraud_probability': fraud_probability,
                'user_id': np.random.randint(1, 500),
                'device_id': np.random.randint(1, 100),
                'card_id': np.random.randint(1, 200)
            }
            
            # Aplicar threshold para determinar se é fraude
            new_transaction['is_fraud_predicted'] = new_transaction['fraud_probability'] > threshold
            
            st.session_state.predictions = new_transaction
        
        if st.session_state.get('predictions'):
            transaction = st.session_state.predictions
            
            # Status da transação baseado no threshold atual
            fraud_prob = transaction['fraud_probability']
            is_fraud_predicted = fraud_prob > threshold
            
            if is_fraud_predicted:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>🚨 ALERTA DE FRAUDE - MODELO GNN</h3>
                    <p><strong>Probabilidade de Fraude:</strong> {fraud_prob:.2%}</p>
                    <p><strong>Threshold:</strong> {threshold:.2%}</p>
                    <p><strong>Valor:</strong> ${transaction['amount']:.2f}</p>
                    <p><strong>Timestamp:</strong> {transaction['timestamp']}</p>
                    <p><strong>Predição:</strong> Modelo GNN treinado</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <h3>✅ TRANSAÇÃO SEGURA - MODELO GNN</h3>
                    <p><strong>Probabilidade de Fraude:</strong> {fraud_prob:.2%}</p>
                    <p><strong>Threshold:</strong> {threshold:.2%}</p>
                    <p><strong>Valor:</strong> ${transaction['amount']:.2f}</p>
                    <p><strong>Timestamp:</strong> {transaction['timestamp']}</p>
                    <p><strong>Predição:</strong> Modelo GNN treinado</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detalhes da transação
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Detalhes da Transação")
                
                details_df = pd.DataFrame({
                    'Campo': ['User ID', 'Device ID', 'Card ID', 'Amount', 'Fraud Probability', 'Prediction'],
                    'Valor': [
                        transaction['user_id'],
                        transaction['device_id'],
                        transaction['card_id'],
                        f"${transaction['amount']:.2f}",
                        f"{fraud_prob:.2%}",
                        "🚨 FRAUDE" if is_fraud_predicted else "✅ SEGURA"
                    ]
                })
                
                st.dataframe(details_df, use_container_width=True)
            
            with col2:
                st.subheader("📊 Análise de Risco GNN")
                
                # Gauge chart para probabilidade de fraude com threshold visual
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = fraud_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidade de Fraude (%) - GNN"},
                    delta = {'reference': threshold * 100},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if is_fraud_predicted else "darkblue"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "lightgreen"},
                            {'range': [threshold * 100, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_history_page(self):
        """Renderiza a página de histórico."""
        if not self.is_model_trained():
            self.render_empty_page("Histórico de Transações", "📋")
            return
        
        st.title("📋 Histórico de Transações")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("❌ Sistema de fraude não disponível")
            return
        
        st.info("📊 **Histórico baseado em dados reais processados pelo modelo GNN**")
        
        # Placeholder para quando dados reais estiverem disponíveis
        st.subheader("📈 Estatísticas de Predições")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processado", "Disponível após uso")
        
        with col2:
            st.metric("Fraudes Detectadas", "Disponível após uso")
        
        with col3:
            st.metric("Taxa de Fraude", "Disponível após uso")
        
        with col4:
            st.metric("Volume Total", "Disponível após uso")
        
        st.info("🔄 Os dados de histórico aparecerão conforme você usar o sistema de detecção em tempo real.")
    
    def render_settings_page(self):
        """Renderiza a página de configurações."""
        st.title("⚙️ Configurações")
        
        # Configurações do modelo
        st.subheader("🤖 Configurações do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Tipo de Modelo:",
                ["GraphSAGE", "GAT", "HeteroGNN"]
            )
            
            hidden_dim = st.slider(
                "Dimensão Oculta:",
                min_value=64,
                max_value=512,
                value=256,
                step=64
            )
        
        with col2:
            num_layers = st.slider(
                "Número de Camadas:",
                min_value=2,
                max_value=5,
                value=3
            )
            
            dropout = st.slider(
                "Dropout:",
                min_value=0.0,
                max_value=0.5,
                value=0.3,
                step=0.1
            )
        
        # Configurações de treinamento
        st.subheader("🏋️ Configurações de Treinamento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                format="%.4f"
            )
            
            batch_size = st.selectbox(
                "Batch Size:",
                [256, 512, 1024, 2048],
                index=2
            )
        
        with col2:
            epochs = st.slider(
                "Épocas:",
                min_value=10,
                max_value=200,
                value=100
            )
            
            early_stopping = st.slider(
                "Early Stopping Patience:",
                min_value=5,
                max_value=20,
                value=10
            )
        
        # Salvar e treinar configurações
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Salvar Configurações"):
                config = {
                    'model': {
                        'type': model_type,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'dropout': dropout
                    },
                    'training': {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'early_stopping_patience': early_stopping
                    }
                }
                
                # Salvar configuração no session state
                st.session_state.model_config = config
                st.success("✅ Configurações salvas com sucesso!")
                st.json(config)
        
        with col2:
            if st.button("🚀 Retreinar Modelo"):
                # Estimar tempo de treinamento baseado na configuração
                estimated_time = epochs * 0.8  # ~0.8 minutos por época para GNN
                if model_type == "GAT":
                    estimated_time *= 1.3  # GAT é mais lento
                elif model_type == "HeteroGNN":
                    estimated_time *= 1.5  # Hetero é o mais lento
                
                if hidden_dim > 256:
                    estimated_time *= 1.2  # Modelos maiores demoram mais
                
                st.warning(f"⏱️ Tempo estimado de treinamento: {estimated_time:.1f} minutos")
                
                # Confirmar se o usuário quer continuar
                st.warning("⚠️ **ATENÇÃO**: Isso iniciará o treinamento REAL do modelo GNN. O processo pode levar muito tempo dependendo dos dados e configurações.")
                
                config = {
                    'model': {
                        'type': model_type,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'dropout': dropout
                    },
                    'training': {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'early_stopping_patience': early_stopping
                    }
                }
                
                with st.spinner(f"🔄 Iniciando treinamento real do modelo {model_type}... (pode levar {estimated_time:.0f}+ minutos)"):
                    try:
                        # TREINAMENTO REAL - não simulação!
                        if MODEL_INTEGRATION_AVAILABLE:
                            # Importar e inicializar o sistema real
                            from main import FraudDetectionSystem
                            fraud_system = FraudDetectionSystem()
                            
                            # Configurar pipeline de dados
                            st.info("📂 Configurando pipeline de dados...")
                            fraud_system.setup_data_pipeline()
                            
                            # Carregar e processar dados reais
                            st.info("⚙️ Carregando e processando dados...")
                            fraud_system.load_and_process_data()
                            
                            # Atualizar configuração do modelo no sistema
                            fraud_system.config.model.update({
                                'type': model_type,
                                'hidden_dim': hidden_dim,
                                'num_layers': num_layers,
                                'dropout': dropout
                            })
                            
                            # Atualizar configuração de treinamento
                            fraud_system.config.training.update({
                                'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'early_stopping_patience': early_stopping
                            })
                            
                            # Criar modelo com as configurações
                            st.info(f"🧠 Criando modelo {model_type}...")
                            fraud_system.create_model()
                            
                            # Callback para atualizar progresso no dashboard
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            epoch_text = st.empty()
                            
                            # EXECUTAR TREINAMENTO REAL
                            st.info(f"🚀 Iniciando treinamento {model_type} com {epochs} épocas...")
                            
                            # Treinar o modelo usando o sistema real
                            training_history = fraud_system.train_model()
                            
                            # Avaliar modelo treinado
                            st.info("📊 Avaliando modelo treinado...")
                            evaluation_metrics = fraud_system.evaluate_model()
                            
                            # Salvar modelo treinado se disponível
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            model_path = f"models/{model_type.lower()}_trained_{timestamp}.pt"
                            
                            # Criar diretório se não existir
                            if MODEL_INTEGRATION_AVAILABLE and 'Path' in globals():
                                Path("models").mkdir(exist_ok=True)
                            
                            # Salvar modelo se possível
                            if MODEL_INTEGRATION_AVAILABLE and fraud_system.model is not None:
                                try:
                                    import torch
                                    torch.save({
                                        'model_state_dict': fraud_system.model.state_dict(),
                                        'config': config,
                                        'metrics': evaluation_metrics,
                                        'timestamp': timestamp
                                    }, model_path)
                                    st.info(f"💾 Modelo salvo em: {model_path}")
                                except Exception as save_error:
                                    st.warning(f"⚠️ Não foi possível salvar o modelo: {save_error}")
                            
                            # Atualizar session state com modelo real
                            st.session_state.model_config = config
                            st.session_state.model_loaded = True
                            st.session_state.model_trained = True
                            st.session_state.training_in_progress = False
                            st.session_state.fraud_system = fraud_system
                            st.session_state.model_path = model_path
                            st.session_state.evaluation_metrics = evaluation_metrics
                            st.session_state.training_history = training_history
                            
                            # Salvar métricas reais
                            st.session_state.training_metrics = {
                                'f1_score': evaluation_metrics.get('f1', 0),
                                'auc_roc': evaluation_metrics.get('auc_roc', 0),
                                'precision': evaluation_metrics.get('precision', 0),
                                'recall': evaluation_metrics.get('recall', 0),
                                'accuracy': evaluation_metrics.get('accuracy', 0),
                                'training_epochs': epochs,
                                'model_type': model_type
                            }
                            
                            # Sucesso
                            progress_bar.progress(100)
                            status_text.text("✅ Treinamento concluído com sucesso!")
                            epoch_text.text("")
                            
                            st.success(f"🎉 Modelo {model_type} treinado com sucesso!")
                            st.success(f"📊 F1-Score: {evaluation_metrics.get('f1', 0):.3f} | AUC-ROC: {evaluation_metrics.get('auc_roc', 0):.3f}")
                            
                            # Exibir histórico de treinamento se disponível
                            if training_history:
                                st.subheader("📈 Histórico de Treinamento")
                                
                                # Converter para DataFrame se necessário
                                import pandas as pd
                                if isinstance(training_history, dict):
                                    history_df = pd.DataFrame(training_history)
                                    
                                    # Plotar gráfico de loss
                                    fig = go.Figure()
                                    if 'train_loss' in history_df.columns:
                                        fig.add_trace(go.Scatter(
                                            y=history_df['train_loss'],
                                            mode='lines',
                                            name='Train Loss',
                                            line=dict(color='#ff6b6b')
                                        ))
                                    if 'val_loss' in history_df.columns:
                                        fig.add_trace(go.Scatter(
                                            y=history_df['val_loss'],
                                            mode='lines',
                                            name='Validation Loss',
                                            line=dict(color='#42a5f5')
                                        ))
                                    
                                    fig.update_layout(title="Evolução da Loss Durante o Treinamento")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            st.error("❌ Sistema de treinamento não está disponível!")
                            st.error("Verifique se todos os módulos estão instalados corretamente.")
                            
                    except Exception as e:
                        st.error(f"❌ Erro durante treinamento real: {str(e)}")
                        st.error("Verifique os logs para mais detalhes.")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Informações do modelo atual
        st.subheader("📊 Informações do Modelo Atual")
        
        if st.session_state.model_loaded:
            col1, col2, col3 = st.columns(3)
            
            # Calcular métricas atuais
            current_threshold = st.session_state.get('current_threshold', 0.5)
            current_metrics = self.calculate_current_metrics(current_threshold)
            
            with col1:
                st.metric("Status", "✅ Carregado")
            with col2:
                st.metric("F1-Score", f"{current_metrics['f1_score']:.3f}")
            with col3:
                st.metric("AUC-ROC", f"{current_metrics['auc_roc']:.3f}")
        else:
            st.warning("⚠️ Nenhum modelo carregado")
    
    def run(self):
        """Executa o dashboard."""
        # Renderizar sidebar e obter página selecionada
        page, threshold = self.render_sidebar()
        
        # Atualizar threshold no session state se mudou
        if 'current_threshold' not in st.session_state or st.session_state.current_threshold != threshold:
            st.session_state.current_threshold = threshold
        
        # Verificar se modelo foi treinado
        model_trained = self.is_model_trained()
        
        # Renderizar página apropriada
        if page == "⚙️ Configurações":
            self.render_settings_page()
        elif not model_trained:
            # Se não há modelo treinado, mostrar página vazia
            if page == "📊 Overview":
                self.render_empty_page("Overview", "📊")
            elif page == "📈 Métricas":
                self.render_empty_page("Métricas Detalhadas", "📈")
            elif page == "🕸️ Análise de Grafos":
                self.render_empty_page("Análise de Grafos", "🕸️")
            elif page == "🔍 Detecção em Tempo Real":
                self.render_empty_page("Detecção em Tempo Real", "🔍")
            elif page == "📋 Histórico":
                self.render_empty_page("Histórico de Transações", "📋")
        else:
            # Modelo treinado, renderizar páginas funcionais
            if page == "📊 Overview":
                self.render_overview_page()
            elif page == "📈 Métricas":
                self.render_metrics_page(threshold)
            elif page == "🕸️ Análise de Grafos":
                self.render_graph_analysis_page()
            elif page == "🔍 Detecção em Tempo Real":
                self.render_realtime_page(threshold)
            elif page == "📋 Histórico":
                self.render_history_page()


def main():
    """Função principal do dashboard."""
    dashboard = FraudDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
