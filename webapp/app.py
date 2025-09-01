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
import networkx as nx

# Add parent directory to path for importing system modules
sys.path.append(str(Path(__file__).parent.parent))

# Detectar disponibilidade de integração sem importar símbolos inutilmente
try:
    import importlib.util as _ilu
    MODEL_INTEGRATION_AVAILABLE = _ilu.find_spec("main") is not None
except Exception as e:
    st.warning(f" Model integration check failed: {e}")
    MODEL_INTEGRATION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="",
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
    
    def is_model_trained(self) -> bool:
        """Retorna True se houver um modelo treinado/carregado no estado da sessão."""
        if 'fraud_system' not in st.session_state or st.session_state.get('fraud_system') is None:
            return False
        return bool(st.session_state.get('model_trained', False) or st.session_state.get('model_loaded', False))

    def render_empty_page(self, title: str, icon: str):
        """Mostra uma página vazia com mensagem para treinar/carregar o modelo."""
        st.title(f"{icon} {title}")
        st.info("Treine e carregue um modelo para ver esta página.")

    def _append_history(self, event: dict):
        """Adiciona um evento ao histórico em memória (session_state)."""
        hist = st.session_state.get('history')
        if hist is None:
            hist = []
        # insere no topo
        hist.insert(0, event)
        st.session_state['history'] = hist

    def render_sidebar(self):
        """Renderiza a sidebar e retorna (page, threshold)."""
        # Páginas disponíveis
        available_pages = [
            " Configurações",
            " Overview",
            " Métricas",
            " Análise de Grafos",
            " Interpretabilidade",
            " Detecção em Tempo Real",
            " Histórico"
        ]

        page = st.sidebar.selectbox("Selecione uma página:", available_pages)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Status do Sistema")

        training_in_progress = st.session_state.get('training_in_progress', False)
        model_trained = self.is_model_trained()

        if training_in_progress:
            st.sidebar.warning(" Treinamento em Progresso...")
        elif model_trained:
            st.sidebar.success(" Modelo GNN Treinado")
            metrics = st.session_state.get('training_metrics') or {}
            if metrics:
                st.sidebar.metric("F1-Score", f"{metrics.get('f1', metrics.get('f1_score', 0)):.3f}")
                st.sidebar.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        else:
            st.sidebar.error(" Nenhum Modelo Treinado")

        if MODEL_INTEGRATION_AVAILABLE:
            st.sidebar.success(" Integração Disponível")
        else:
            st.sidebar.error(" Integração Indisponível")

        st.sidebar.markdown("---")

        # Configurações rápidas
        model_trained = self.is_model_trained()
        if model_trained:
            st.sidebar.subheader("Configurações Rápidas")
            current_thr = float(st.session_state.get('current_threshold', 0.5))
            threshold = st.sidebar.slider(
                "Threshold de Fraude:",
                min_value=0.0,
                max_value=1.0,
                value=current_thr,
                step=0.01
            )
            auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
            if auto_refresh:
                st.sidebar.info(" Auto-refresh ativo")
        else:
            threshold = 0.5

        return page, threshold
    
    def render_metrics_page(self, threshold=0.5):
        """Renderiza a página de métricas detalhadas."""
        if not self.is_model_trained():
            self.render_empty_page("Métricas Detalhadas", "")
            return

        st.title(" Métricas Detalhadas")

        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error(" Sistema de fraude não disponível")
            return

        # Controle de threshold local (sincronizado com a sidebar)
        col_thr1, col_thr2 = st.columns([3, 1])
        with col_thr1:
            thr_local = st.slider(
                "Threshold para classificar fraude",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get('current_threshold', threshold)),
                step=0.01
            )
        with col_thr2:
            if st.button("Aplicar threshold"):
                st.session_state.current_threshold = thr_local
                st.rerun()
        threshold = float(st.session_state.get('current_threshold', thr_local))
        st.info(f" Threshold em uso: {threshold:.2f}")

        # Mostrar métricas do modelo real
        evaluation_metrics = st.session_state.get('evaluation_metrics', {})

        # Recalcular métricas com base no threshold usando dados de validação
        y_true, y_scores, y_pred = self._get_validation_outputs(fraud_system, threshold)
        if y_true is not None and y_pred is not None:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            except Exception:
                acc = evaluation_metrics.get('accuracy', 0)
                prec = evaluation_metrics.get('precision', 0)
                rec = evaluation_metrics.get('recall', 0)
                f1 = evaluation_metrics.get('f1', 0)
        else:
            acc = evaluation_metrics.get('accuracy', 0)
            prec = evaluation_metrics.get('precision', 0)
            rec = evaluation_metrics.get('recall', 0)
            f1 = evaluation_metrics.get('f1', 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("Precision", f"{prec:.3f}")
        with col3:
            st.metric("Recall", f"{rec:.3f}")
        with col4:
            st.metric("F1-Score", f"{f1:.3f}")

        # Contagem de positivos/negativos previstos na validação
        if y_true is not None and y_pred is not None:
            try:
                import numpy as np
                pos_count = int(np.sum(y_pred == 1))
                neg_count = int(np.sum(y_pred == 0))
                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    st.metric("Pred. Pos.", f"{pos_count}")
                with colp2:
                    st.metric("Pred. Neg.", f"{neg_count}")
                with colp3:
                    st.metric("Amostra (val)", f"{len(y_true)}")
            except Exception:
                pass

        # Exibir informações do modelo treinado
        st.subheader("Informações do Modelo Treinado")
        training_metrics = st.session_state.get('training_metrics', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tipo de Modelo", training_metrics.get('model_type', 'N/A'))

        with col2:
            st.metric("Épocas Treinadas", training_metrics.get('training_epochs', 0))

        with col3:
            # AUC pode ser recalculada com base nos scores atuais
            try:
                from sklearn.metrics import roc_auc_score
                auc_val = roc_auc_score(y_true, y_scores) if (y_true is not None and y_scores is not None) else evaluation_metrics.get('auc_roc', 0)
            except Exception:
                auc_val = evaluation_metrics.get('auc_roc', 0)
            st.metric("AUC-ROC", f"{auc_val:.3f}")

        # Curva ROC
        if y_true is not None and y_scores is not None:
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#1f77b4')))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(color='gray', dash='dash')))
                fig_roc.update_layout(title='Curva ROC', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
                st.plotly_chart(fig_roc, width='stretch')
            except Exception as e:
                st.warning(f"Não foi possível gerar a curva ROC: {e}")

        # Matriz de confusão
        if y_true is not None and y_pred is not None:
            try:
                from sklearn.metrics import confusion_matrix
                import numpy as np
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = (0, 0, 0, 0)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                st.subheader("Matriz de Confusão")
                col_cm1, col_cm2 = st.columns([2, 1])
                with col_cm1:
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Pred. Neg', 'Pred. Pos'],
                        y=['Real Neg', 'Real Pos'],
                        colorscale='Blues',
                        showscale=True,
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 14}
                    ))
                    fig_cm.update_layout(height=350)
                    st.plotly_chart(fig_cm, width='stretch')
                with col_cm2:
                    st.metric("TP", f"{tp}")
                    st.metric("TN", f"{tn}")
                    st.metric("FP", f"{fp}")
                    st.metric("FN", f"{fn}")
            except Exception as e:
                st.warning(f"Não foi possível calcular a matriz de confusão: {e}")

        # Histograma de scores com linha do threshold
        if y_true is not None and y_scores is not None:
            try:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=y_scores, nbinsx=30, name='Scores de fraude', marker_color='#1f77b4', opacity=0.75))
                fig_hist.add_shape(type='line', x0=threshold, x1=threshold, y0=0, y1=1, yref='paper', line=dict(color='red', dash='dash'))
                fig_hist.update_layout(title='Distribuição de Scores (validação) com Threshold', xaxis_title='Score de fraude (prob)', yaxis_title='Contagem', barmode='overlay', height=350)
                st.plotly_chart(fig_hist, width='stretch')
            except Exception:
                pass

        # Histórico de treinamento se disponível
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("Evolução do Treinamento")

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

                st.plotly_chart(fig, width='stretch')
    
    def render_overview_page(self):
        """Renderiza a página de overview."""
        if not self.is_model_trained():
            self.render_empty_page("Overview", "")
            return
        
        st.title("Fraud Detection - Overview")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error("Sistema de fraude não disponível")
            return
        
        # Calcular métricas atuais do modelo real
        evaluation_metrics = st.session_state.get('evaluation_metrics', {})
        
        # KPIs principais do modelo treinado
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Accuracy",
                value=f"{evaluation_metrics.get('accuracy', 0):.2%}",
                delta="+Real"
            )
        
        with col2:
            st.metric(
                label="Precision",
                value=f"{evaluation_metrics.get('precision', 0):.2%}",
                delta="+Real"
            )
        
        with col3:
            st.metric(
                label="Recall",
                value=f"{evaluation_metrics.get('recall', 0):.2%}",
                delta="+Real"
            )
        
        with col4:
            st.metric(
                label="F1-Score",
                value=f"{evaluation_metrics.get('f1', 0):.2%}",
                delta="+Real"
            )
        
        st.markdown("---")
        
        # Histórico de treinamento real
        if 'training_history' in st.session_state and st.session_state.training_history:
            st.subheader("Histórico de Treinamento Real")
            
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
            
            st.plotly_chart(fig, width='stretch')
        
        # Informações do modelo
        st.subheader("Informações do Modelo")
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
    
    def render_metrics_page_v2(self, threshold=0.5):
        """Versão desativada/antiga da página de métricas (mantida para compatibilidade)."""
        # Esta função está intencionalmente vazia para evitar conflitos.
        pass
    
    def render_graph_analysis_page(self):
        """Renderiza a página de análise de grafos."""
        if not self.is_model_trained():
            self.render_empty_page("Análise de Grafos", "")
            return
        
        st.title(" Análise de Grafos")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error(" Sistema de fraude não disponível")
            return
        
        st.info("**Análise de grafos baseada no modelo GNN treinado**")

        graph_summary = st.session_state.get('graph_summary')
        train_graph = getattr(fraud_system, 'train_graph', None)

        st.subheader("Estatísticas do Grafo Treinado")
        if graph_summary:
            # Métricas básicas
            num_nodes = graph_summary.get('train_graph', {}).get('num_transaction_nodes', 0)
            num_features = graph_summary.get('train_graph', {}).get('num_features', 0)
            fraud_rate = graph_summary.get('train_graph', {}).get('fraud_rate', 0.0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transações (nós)", f"{num_nodes}")
            with col2:
                st.metric("Features/Transação", f"{num_features}")
            with col3:
                st.metric("Taxa de Fraude", f"{fraud_rate:.2%}")

            st.markdown("---")
        else:
            st.warning("Sem resumo do grafo em memória. Treine o modelo novamente.")

    # Estatísticas por tipo (se grafo estiver disponível)
        if train_graph is not None:
            try:
                # Contagem de nós por tipo
                node_counts = {nt: int(getattr(train_graph[nt], 'num_nodes', 0)) for nt in train_graph.node_types}
                # Contagem de arestas por tipo
                edge_counts = {str(et): int(train_graph[et].edge_index.shape[1]) for et in train_graph.edge_types}

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Nós por tipo")
                    st.dataframe(
                        pd.DataFrame([
                            {"Tipo": k, "Quantidade": v} for k, v in node_counts.items()
                        ]),
                        width='stretch'
                    )
                with col2:
                    st.write("Arestas por tipo")
                    st.dataframe(
                        pd.DataFrame([
                            {"Tipo": k, "Quantidade": v} for k, v in edge_counts.items()
                        ]),
                        width='stretch'
                    )

                # Pequena análise de graus dos nós de transação
                st.subheader("Distribuição de Grau (Transações)")
                import numpy as np
                deg_counts = []
                # Considerar arestas que chegam/saem de transações
                for et in train_graph.edge_types:
                    src, rel, dst = et
                    ei = train_graph[et].edge_index
                    if dst == 'transaction':
                        deg = np.bincount(ei[1].cpu().numpy(), minlength=train_graph['transaction'].num_nodes)
                        deg_counts.append(deg)
                    if src == 'transaction':
                        deg = np.bincount(ei[0].cpu().numpy(), minlength=train_graph['transaction'].num_nodes)
                        deg_counts.append(deg)
                if deg_counts:
                    total_deg = np.sum(deg_counts, axis=0)
                    hist = np.bincount(total_deg)
                    df_hist = pd.DataFrame({"grau": np.arange(len(hist)), "contagem": hist})
                    st.bar_chart(df_hist.set_index('grau'))
            except Exception as e:
                st.warning(f"Não foi possível calcular estatísticas do grafo: {e}")
        else:
            st.info("Grafo de treino não disponível na sessão.")

        st.markdown("---")
        st.subheader("Visualização do Grafo (amostra)")
        sample_n = st.slider("Número de nós de transação na amostra", min_value=30, max_value=400, value=150, step=10)
        if train_graph is not None:
            try:
                fig_graph = self._build_graph_figure(train_graph, sample_n)
                st.plotly_chart(fig_graph, width='stretch')
            except Exception as e:
                st.warning(f"Não foi possível gerar a visualização do grafo: {e}")

        st.markdown("---")
        st.subheader(" Usuários mais 'fraudadores'")
        with st.expander("Métricas agregadas por usuário", expanded=True):
            colu1, colu2, colu3 = st.columns([2,2,2])
            with colu1:
                top_k = st.slider("Top K usuários", min_value=5, max_value=50, value=10, step=1)
            with colu2:
                min_tx = st.slider("Mínimo de transações por usuário", min_value=1, max_value=50, value=5, step=1)
            with colu3:
                sort_by = st.selectbox("Ordenar por", options=["fraud_rate", "pred_rate", "fraud_count", "pred_count", "avg_prob"], index=0)

            try:
                df_users = self._compute_user_fraud_metrics(fraud_system, float(st.session_state.get('current_threshold', 0.5)))
                if df_users is None or df_users.empty:
                    st.info("Não foi possível calcular métricas por usuário.")
                else:
                    # Filtrar por mínimo de transações
                    dfu = df_users[df_users['total_tx'] >= int(min_tx)].copy()
                    if dfu.empty:
                        st.info("Nenhum usuário com pelo menos o mínimo de transações definido.")
                    else:
                        # Garantir colunas presentes para ordenação
                        valid_sort = sort_by if sort_by in dfu.columns else 'fraud_rate'
                        dfu_sorted = dfu.sort_values(valid_sort, ascending=False).head(int(top_k))
                        st.dataframe(dfu_sorted.reset_index(drop=True), width='stretch')

                        # Gráfico de barras
                        metric_plot = valid_sort
                        figu = go.Figure()
                        figu.add_trace(go.Bar(x=dfu_sorted['user_id'].astype(str), y=dfu_sorted[metric_plot], marker_color="#d62728" if 'fraud' in metric_plot or 'pred' in metric_plot else "#1f77b4"))
                        figu.update_layout(title=f"Top {len(dfu_sorted)} usuários por {metric_plot}", xaxis_title="user_id", yaxis_title=metric_plot, height=350)
                        st.plotly_chart(figu, width='stretch')
            except Exception as e:
                st.warning(f"Falha ao calcular métricas por usuário: {e}")
    
    def render_realtime_page(self, threshold=0.5):
        """Renderiza a página de detecção em tempo real."""
        if not self.is_model_trained():
            self.render_empty_page("Detecção em Tempo Real", "")
            return
        
        st.title("Detecção em Tempo Real")
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error(" Sistema de fraude não disponível")
            return

        st.info(f" **Threshold Atual:** {threshold:.2f}")
        st.info(" **Usando modelo GNN treinado para predições reais**")
        
        st.subheader(" Monitor de Transações")

        if 'history' not in st.session_state:
            st.session_state.history = []

        # Simulação aleatória (mantida), usa modelo real por índice aleatório
        if st.button(" Simular Nova Transação"):
            st.info(" Processando transação com modelo GNN...")
            fraud_probability = self._predict_random_transaction_probability(st.session_state.get('fraud_system'), split='val')
            if fraud_probability is None:
                st.error("Não foi possível calcular probabilidade com o modelo. Verifique se o modelo está treinado e o grafo possui máscara de validação.")
                st.stop()
            new_transaction = {
                'timestamp': datetime.now(),
                'amount': float(np.random.lognormal(3, 1.5)),
                'fraud_probability': float(fraud_probability),
                'user_id': int(np.random.randint(1, 500)),
                'device_id': int(np.random.randint(1, 100)),
                'card_id': int(np.random.randint(1, 200))
            }
            new_transaction['is_fraud_predicted'] = new_transaction['fraud_probability'] > threshold
            st.session_state.predictions = new_transaction
            self._append_history({
                'timestamp': new_transaction['timestamp'],
                'amount': new_transaction['amount'],
                'prob': new_transaction['fraud_probability'],
                'pred': new_transaction['is_fraud_predicted'],
                'user_id': new_transaction['user_id'],
                'device_id': new_transaction['device_id'],
                'card_id': new_transaction['card_id'],
                'method': 'simulado',
                'threshold': threshold,
            })

        # ===== Transação manual: opções vindas do grafo + inferência real sem fallback =====
        with st.expander(" Criar transação manualmente (GNN)", expanded=True):
            data = getattr(fraud_system, 'train_graph', None)
            if data is None:
                st.warning("Grafo de treino indisponível.")
            else:
                # Opções válidas de IDs conforme o grafo
                def _num_nodes(nt):
                    n = int(getattr(data[nt], 'num_nodes', 0))
                    if n == 0 and hasattr(data[nt], 'x') and data[nt].x is not None:
                        n = int(data[nt].x.shape[0])
                    return n

                n_users = _num_nodes('user')
                n_devices = _num_nodes('device')
                n_cards = _num_nodes('card')

                with st.form(key="manual_tx_form"):
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        amount = st.number_input("Valor (amount)", min_value=0.0, value=100.0, step=10.0)
                    with mcol2:
                        user_id = st.selectbox("User ID", options=list(range(max(0, n_users))), index=0)
                    with mcol3:
                        device_id = st.selectbox("Device ID", options=list(range(max(0, n_devices))), index=0)
                    with mcol4:
                        card_id = st.selectbox("Card ID", options=list(range(max(0, n_cards))), index=0)

                    submitted = st.form_submit_button(" Avaliar Transação Manual (GNN)")

                    if submitted:
                        try:
                            is_fraud, prob, new_idx = self._predict_manual_transaction(
                                fraud_system=fraud_system,
                                amount=float(amount),
                                user_id=int(user_id),
                                device_id=int(device_id),
                                card_id=int(card_id),
                                threshold=float(threshold),
                            )

                            manual_tx = {
                                'timestamp': datetime.now(),
                                'amount': float(amount),
                                'fraud_probability': float(prob),
                                'user_id': int(user_id),
                                'device_id': int(device_id),
                                'card_id': int(card_id),
                            }
                            manual_tx['is_fraud_predicted'] = bool(is_fraud)

                            st.session_state.predictions = manual_tx
                            # Guardar índice do novo nó para interpretabilidade
                            st.session_state['last_manual_new_tx_idx'] = int(new_idx)
                            # Guardar contexto da última transação manual (para interpretabilidade reconstruída)
                            st.session_state['last_manual_ctx'] = {
                                'amount': float(amount),
                                'user_id': int(user_id),
                                'device_id': int(device_id),
                                'card_id': int(card_id)
                            }
                            self._append_history({
                                'timestamp': manual_tx['timestamp'],
                                'amount': manual_tx['amount'],
                                'prob': manual_tx['fraud_probability'],
                                'pred': manual_tx['is_fraud_predicted'],
                                'user_id': manual_tx['user_id'],
                                'device_id': manual_tx['device_id'],
                                'card_id': manual_tx['card_id'],
                                'method': 'manual_gnn',
                                'threshold': threshold,
                            })

                            st.success(f"Probabilidade (classe fraude): {prob:.4f}")
                            st.info("Predição: " + (" FRAUDE" if is_fraud else " SEGURA"))

                            # Visualização do novo nó conectado
                            fig_new = self._build_manual_tx_figure(user_id=int(user_id), device_id=int(device_id), card_id=int(card_id))
                            st.plotly_chart(fig_new, width='stretch')

                        except Exception as e:
                            st.error(f"Falha na inferência da transação manual: {e}")

        if st.session_state.get('predictions'):
            transaction = st.session_state.predictions
            
            # Status da transação baseado no threshold atual
            fraud_prob = transaction['fraud_probability']
            is_fraud_predicted = fraud_prob > threshold
            
            if is_fraud_predicted:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3> ALERTA DE FRAUDE - MODELO GNN</h3>
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
                    <h3> TRANSAÇÃO SEGURA - MODELO GNN</h3>
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
                st.subheader(" Detalhes da Transação")
                
                details_df = pd.DataFrame({
                    'Campo': ['User ID', 'Device ID', 'Card ID', 'Amount', 'Fraud Probability', 'Prediction'],
                    'Valor': [
                        transaction['user_id'],
                        transaction['device_id'],
                        transaction['card_id'],
                        f"${transaction['amount']:.2f}",
                        f"{fraud_prob:.2%}",
                        " FRAUDE" if is_fraud_predicted else " SEGURA"
                    ]
                })
                
                st.dataframe(details_df, width='stretch')
            
            with col2:
                st.subheader(" Análise de Risco GNN")
                
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
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.subheader(" Últimos eventos")
        history = st.session_state.get('history') or []
        if history:
            df_hist = pd.DataFrame(history)
            # Renomear colunas para visualização
            df_show = df_hist.rename(columns={
                'timestamp': 'Timestamp',
                'amount': 'Valor',
                'prob': 'Prob_Fraude',
                'pred': 'Predição',
                'method': 'Método',
                'threshold': 'Threshold',
            })
            st.dataframe(df_show.head(50), width='stretch')
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(" Baixar histórico (CSV)", data=csv, file_name="historico_transacoes.csv", mime="text/csv")
        else:
            st.info("Sem eventos ainda. Simule ou crie uma transação manual.")
    
    def render_history_page(self):
        """Renderiza a página de histórico."""
        if not self.is_model_trained():
            self.render_empty_page("Histórico de Transações", "")
            return
        
        st.title(" Histórico de Transações")
        
        # Usar dados reais do modelo treinado
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error(" Sistema de fraude não disponível")
            return
        
        st.info(" Histórico gerado a partir dos eventos processados na aba de Tempo Real.")

        history = st.session_state.get('history') or []
        if history:
            df_hist = pd.DataFrame(history)
            total = len(df_hist)
            fraudes = int(df_hist['pred'].sum()) if 'pred' in df_hist else 0
            taxa = fraudes / total if total > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processado", f"{total}")
            with col2:
                st.metric("Fraudes Detectadas", f"{fraudes}")
            with col3:
                st.metric("Taxa de Fraude", f"{taxa:.2%}")

            st.subheader(" Eventos")
            st.dataframe(df_hist, width='stretch')
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(" Baixar histórico (CSV)", data=csv, file_name="historico_transacoes.csv", mime="text/csv")
        else:
            st.info("Sem eventos no histórico.")

        st.info(" Os dados de histórico aparecerão conforme você usar o sistema de detecção em tempo real.")
    
    def render_settings_page(self):
        """Renderiza a página de configurações."""
        st.title(" Configurações")
        
        # Configurações de dados
        st.subheader("🗃️ Configurações de Dados")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            sample_size = st.number_input(
                "Tamanho da amostra (0 = completo)",
                min_value=0,
                max_value=1_000_000,
                value=int(st.session_state.get('model_config', {}).get('data', {}).get('sample_size', 50000)),
                step=1000
            )
        with dcol2:
            val_split = st.slider(
                "Proporção de Validação",
                min_value=0.0,
                max_value=0.5,
                value=float(st.session_state.get('model_config', {}).get('data', {}).get('val_split', 0.1)),
                step=0.01
            )
        with dcol3:
            test_split = st.slider(
                "Proporção de Teste",
                min_value=0.0,
                max_value=0.8,
                value=float(st.session_state.get('model_config', {}).get('data', {}).get('test_split', 0.2)),
                step=0.01
            )
        train_split = max(0.0, 1.0 - (val_split + test_split))
        if train_split <= 0:
            st.error("A soma de validação e teste deve ser menor que 1. Ajuste os sliders.")
        st.caption(f"Divisão final: treino ≈ {train_split:.2f}, validação = {val_split:.2f}, teste = {test_split:.2f}")

        # Configurações do modelo
        st.subheader(" Configurações do Modelo")
        
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
        st.subheader(" Configurações de Treinamento")
        
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
            if st.button(" Salvar Configurações"):
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
                    },
                    'data': {
                        'sample_size': int(sample_size) if int(sample_size) > 0 else None,
                        'val_split': float(val_split),
                        'test_split': float(test_split)
                    }
                }
                
                # Salvar configuração no session state
                st.session_state.model_config = config
                st.success(" Configurações salvas com sucesso!")
                st.json(config)
        
        with col2:
            if st.button(" Retreinar Modelo"):
                # Estimar tempo de treinamento baseado na configuração
                estimated_time = epochs * 0.8  # ~0.8 minutos por época para GNN
                if model_type == "GAT":
                    estimated_time *= 1.3  # GAT é mais lento
                elif model_type == "HeteroGNN":
                    estimated_time *= 1.5  # Hetero é o mais lento
                
                if hidden_dim > 256:
                    estimated_time *= 1.2  # Modelos maiores demoram mais
                
                st.warning(f" Tempo estimado de treinamento: {estimated_time:.1f} minutos")
                
                # Confirmar se o usuário quer continuar
                st.warning(" **ATENÇÃO**: Isso iniciará o treinamento REAL do modelo GNN. O processo pode levar muito tempo dependendo dos dados e configurações.")
                
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
                    },
                    'data': {
                        'sample_size': int(sample_size) if int(sample_size) > 0 else None,
                        'val_split': float(val_split),
                        'test_split': float(test_split)
                    }
                }
                
                # Marcar treinamento em progresso para liberar páginas na navegação
                st.session_state.training_in_progress = True

                with st.spinner(f" Iniciando treinamento real do modelo {model_type}... (pode levar {estimated_time:.0f}+ minutos)"):
                    try:
                        # TREINAMENTO REAL - não simulação!
                        if MODEL_INTEGRATION_AVAILABLE:
                            # Importar e inicializar o sistema real
                            from main import FraudDetectionSystem
                            fraud_system = FraudDetectionSystem()
                            
                            # Atualizar configuração de dados ANTES de criar a pipeline
                            fraud_system.config.set('data.sample_size', (int(sample_size) if int(sample_size) > 0 else None))
                            fraud_system.config.set('data.val_split', float(val_split))
                            fraud_system.config.set('data.test_split', float(test_split))
                            
                            # Configurar pipeline de dados com os splits desejados
                            st.info(" Configurando pipeline de dados...")
                            fraud_system.setup_data_pipeline()
                            
                            # Carregar e processar dados reais
                            st.info(" Carregando e processando dados...")
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
                            st.info(f" Criando modelo {model_type}...")
                            fraud_system.create_model()
                            
                            # Callback para atualizar progresso no dashboard
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            epoch_text = st.empty()
                            
                            # EXECUTAR TREINAMENTO REAL
                            st.info(f" Iniciando treinamento {model_type} com {epochs} épocas...")
                            
                            # Treinar o modelo usando o sistema real
                            training_history = fraud_system.train_model()
                            
                            # Avaliar modelo treinado
                            st.info(" Avaliando modelo treinado...")
                            evaluation_metrics = fraud_system.evaluate_model()
                            
                            # Resumo do grafo para a página de análise, se disponível
                            graph_summary = None
                            try:
                                dp = getattr(fraud_system, 'data_pipeline', None)
                                if dp is not None and hasattr(dp, 'get_data_summary'):
                                    graph_summary = dp.get_data_summary()
                            except Exception:
                                graph_summary = None
                            
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
                                    st.info(f" Modelo salvo em: {model_path}")
                                except Exception as save_error:
                                    st.warning(f" Não foi possível salvar o modelo: {save_error}")
                            
                            # Atualizar session state com modelo real
                            st.session_state.model_config = config
                            st.session_state.model_loaded = True
                            st.session_state.model_trained = True
                            st.session_state.training_in_progress = False
                            st.session_state.fraud_system = fraud_system
                            st.session_state.model_path = model_path
                            st.session_state.evaluation_metrics = evaluation_metrics
                            st.session_state.training_history = training_history
                            st.session_state.graph_summary = graph_summary
                            
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
                            status_text.text(" Treinamento concluído com sucesso!")
                            epoch_text.text("")
                            
                            st.success(f"🎉 Modelo {model_type} treinado com sucesso!")
                            st.success(f" F1-Score: {evaluation_metrics.get('f1', 0):.3f} | AUC-ROC: {evaluation_metrics.get('auc_roc', 0):.3f}")
                            
                            # Exibir histórico de treinamento se disponível
                            if training_history:
                                st.subheader(" Histórico de Treinamento")
                                
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
                                    st.plotly_chart(fig, width='stretch')
                            
                        else:
                            st.error(" Sistema de treinamento não está disponível!")
                            st.error("Verifique se todos os módulos estão instalados corretamente.")
                            
                    except Exception as e:
                        st.error(f" Erro durante treinamento real: {str(e)}")
                        st.error("Verifique os logs para mais detalhes.")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        # Forçar refresh da UI para liberar as páginas
                        try:
                            st.rerun()
                        except Exception:
                            pass
    
    def render_model_info_section(self):
        """Seção com informações do modelo atual (status e métricas principais)."""
        st.subheader(" Informações do Modelo Atual")
        if st.session_state.get('model_loaded', False):
            col1, col2, col3 = st.columns(3)
            current_metrics = st.session_state.get('evaluation_metrics') or {}
            with col1:
                st.metric("Status", " Carregado")
            with col2:
                st.metric("F1-Score", f"{current_metrics.get('f1', 0):.3f}")
            with col3:
                st.metric("AUC-ROC", f"{current_metrics.get('auc_roc', 0):.3f}")
        else:
            st.warning(" Nenhum modelo carregado")

    def _get_validation_outputs(self, fraud_system, threshold: float):
        """Obtém y_true, y_scores e y_pred na partição de validação do grafo de treino."""
        try:
            model = fraud_system.model
            data = fraud_system.train_graph
            model.eval()
            import torch
            with torch.no_grad():
                if hasattr(data, 'x_dict'):
                    logits = model(data.x_dict, data.edge_index_dict)
                    labels = data['transaction'].y
                    val_mask = data['transaction'].val_mask
                    logits = logits[val_mask]
                    y_true = labels[val_mask].cpu().numpy()
                else:
                    logits = model(data.x, data.edge_index)
                    labels = data.y
                    val_mask = data.val_mask
                    logits = logits[val_mask]
                    y_true = labels[val_mask].cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_pred = (probs > threshold).astype(int)
                return y_true, probs, y_pred
        except Exception:
            return None, None, None

    def _predict_random_transaction_probability(self, fraud_system, split: str = 'val') -> float | None:
        """Seleciona aleatoriamente uma transação do split indicado e retorna a probabilidade de fraude.
        split: 'train' | 'val' | 'test'
        Retorna None se não for possível calcular.
        """
        try:
            if fraud_system is None or getattr(fraud_system, 'model', None) is None:
                return None
            model = fraud_system.model
            data = fraud_system.train_graph
            if data is None:
                return None

            # Selecionar máscara
            if hasattr(data, 'x_dict'):
                if split == 'train':
                    mask = data['transaction'].train_mask
                elif split == 'test' and hasattr(data['transaction'], 'test_mask'):
                    mask = data['transaction'].test_mask
                else:
                    mask = data['transaction'].val_mask
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    return None
                import torch
                ridx = idx[torch.randint(0, idx.numel(), (1,)).item()]
                model.eval()
                with torch.no_grad():
                    logits = model(data.x_dict, data.edge_index_dict)
                    prob = torch.softmax(logits[ridx], dim=0)[1].item()
                return float(prob)
            else:
                if split == 'train':
                    mask = data.train_mask
                elif split == 'test' and hasattr(data, 'test_mask'):
                    mask = data.test_mask
                else:
                    mask = data.val_mask
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    return None
                import torch
                ridx = idx[torch.randint(0, idx.numel(), (1,)).item()]
                model.eval()
                with torch.no_grad():
                    logits = model(data.x, data.edge_index)
                    prob = torch.softmax(logits[ridx], dim=0)[1].item()
                return float(prob)
        except Exception:
            return None

    def _predict_by_index(self, fraud_system, index: int, split: str | None = None):
        """Prediz fraude (bool) e probabilidade para um nó de transação específico.
        Se 'split' for dado, valida que o índice pertence à máscara correspondente.
        Retorna tuple (is_fraud: bool, prob: float) ou None em caso de falha/índice inválido.
        """
        try:
            if fraud_system is None or getattr(fraud_system, 'model', None) is None:
                return None
            model = fraud_system.model
            data = fraud_system.train_graph
            if data is None:
                return None
            import torch
            model.eval()
            with torch.no_grad():
                if hasattr(data, 'x_dict'):
                    # Validar máscara por split
                    mask = None
                    if split == 'train' and hasattr(data['transaction'], 'train_mask'):
                        mask = data['transaction'].train_mask
                    elif split == 'val' and hasattr(data['transaction'], 'val_mask'):
                        mask = data['transaction'].val_mask
                    elif split == 'test' and hasattr(data['transaction'], 'test_mask'):
                        mask = data['transaction'].test_mask
                    if mask is not None:
                        valid_idx_t = mask.nonzero(as_tuple=True)[0]
                        valid_idx = set(valid_idx_t.cpu().numpy().tolist())
                        if index not in valid_idx:
                            return None
                    logits = model(data.x_dict, data.edge_index_dict)
                    p = torch.softmax(logits[index], dim=0)[1].item()
                else:
                    mask = None
                    if split == 'train' and hasattr(data, 'train_mask'):
                        mask = data.train_mask
                    elif split == 'val' and hasattr(data, 'val_mask'):
                        mask = data.val_mask
                    elif split == 'test' and hasattr(data, 'test_mask'):
                        mask = data.test_mask
                    if mask is not None:
                        valid_idx_t = mask.nonzero(as_tuple=True)[0]
                        valid_idx = set(valid_idx_t.cpu().numpy().tolist())
                        if index not in valid_idx:
                            return None
                    logits = model(data.x, data.edge_index)
                    p = torch.softmax(logits[index], dim=0)[1].item()
                return (p > st.session_state.get('current_threshold', 0.5), float(p))
        except Exception:
            return None

    def _build_graph_figure(self, hetero_data, sample_n: int = 150):
        """Gera figura Plotly de uma amostra do grafo heterogêneo."""
        import numpy as np
        G = nx.Graph()
        num_tx = int(hetero_data['transaction'].num_nodes)
        sample_n = max(10, min(sample_n, num_tx))

        # Amostrar: priorizar fraudes
        y = hetero_data['transaction'].y.cpu().numpy() if hasattr(hetero_data['transaction'], 'y') else np.zeros(num_tx)
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]
        pick_fraud = fraud_idx[: min(len(fraud_idx), sample_n // 3)]
        remaining = sample_n - len(pick_fraud)
        if remaining > 0:
            rng = np.random.default_rng(42)
            pick_legit = rng.choice(legit_idx, size=min(remaining, len(legit_idx)), replace=False)
        else:
            pick_legit = np.array([], dtype=int)
        tx_sample = np.unique(np.concatenate([pick_fraud, pick_legit]))

        # Adicionar nós de transação
        for t in tx_sample:
            G.add_node(f"t_{t}", tipo='transaction', fraud=int(y[t]))

        # Mapear vizinhos por aresta
        def add_edge_u_t(ei):
            ei_np = ei.cpu().numpy()
            for u, t in ei_np.T:
                if t in tx_sample:
                    G.add_node(f"u_{u}", tipo='user')
                    G.add_edge(f"u_{u}", f"t_{t}")

        def add_edge_t_c(ei):
            ei_np = ei.cpu().numpy()
            for t, c in ei_np.T:
                if t in tx_sample:
                    G.add_node(f"c_{c}", tipo='card')
                    G.add_edge(f"t_{t}", f"c_{c}")

        def add_edge_t_d(ei):
            ei_np = ei.cpu().numpy()
            for t, d in ei_np.T:
                if t in tx_sample:
                    G.add_node(f"d_{d}", tipo='device')
                    G.add_edge(f"t_{t}", f"d_{d}")

        for et in hetero_data.edge_types:
            src, rel, dst = et
            ei = hetero_data[et].edge_index
            if (src, dst) == ('user', 'transaction'):
                add_edge_u_t(ei)
            elif (src, dst) == ('transaction', 'card'):
                add_edge_t_c(ei)
            elif (src, dst) == ('transaction', 'device'):
                add_edge_t_d(ei)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.15)

        # Traces por tipo de nó
        tipo_cores = {
            'transaction_0': '#1f77b4',  # azul
            'transaction_1': '#d62728',  # vermelho
            'user': '#2ca02c',           # verde
            'card': '#9467bd',           # roxo
            'device': '#ff7f0e'          # laranja
        }

        def make_node_trace(nodes_ids):
            xs, ys, texts, colors = [], [], [], []
            for n in nodes_ids:
                x, y0 = pos[n]
                xs.append(x)
                ys.append(y0)
                attrs = G.nodes[n]
                tipo = attrs.get('tipo')
                if tipo == 'transaction':
                    fraud = int(attrs.get('fraud', 0))
                    colors.append(tipo_cores[f'transaction_{fraud}'])
                    texts.append(f"Transação {n.split('_')[1]} - {'Fraude' if fraud==1 else 'Legítima'}")
                else:
                    colors.append(tipo_cores[tipo])
                    prefixo = {'user':'Usuário','card':'Cartão','device':'Dispositivo'}[tipo]
                    texts.append(f"{prefixo} {n.split('_')[1]}")
            return go.Scatter(x=xs, y=ys, mode='markers', text=texts, hoverinfo='text',
                               marker=dict(size=8, color=colors, line=dict(width=0.5, color='#333')))

        # Edges trace
        xe, ye = [], []
        for (u, v) in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xe += [x0, x1, None]
            ye += [y0, y1, None]
        edge_trace = go.Scatter(x=xe, y=ye, mode='lines', line=dict(color='#aaa', width=1), hoverinfo='none')

        # Nodes por tipo
        nodes_t0 = [n for n, a in G.nodes(data=True) if a.get('tipo')=='transaction' and int(a.get('fraud',0))==0]
        nodes_t1 = [n for n, a in G.nodes(data=True) if a.get('tipo')=='transaction' and int(a.get('fraud',0))==1]
        nodes_u = [n for n, a in G.nodes(data=True) if a.get('tipo')=='user']
        nodes_c = [n for n, a in G.nodes(data=True) if a.get('tipo')=='card']
        nodes_d = [n for n, a in G.nodes(data=True) if a.get('tipo')=='device']

        fig = go.Figure()
        fig.add_trace(edge_trace)
        if nodes_t0:
            fig.add_trace(make_node_trace(nodes_t0).update(name='Transação Legítima'))
        if nodes_t1:
            fig.add_trace(make_node_trace(nodes_t1).update(name='Transação Fraude'))
        if nodes_u:
            fig.add_trace(make_node_trace(nodes_u).update(name='Usuário'))
        if nodes_c:
            fig.add_trace(make_node_trace(nodes_c).update(name='Cartão'))
        if nodes_d:
            fig.add_trace(make_node_trace(nodes_d).update(name='Dispositivo'))

        fig.update_layout(
            title='Amostra do Grafo (tipos de nós e conexões)',
            showlegend=True,
            height=650,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def _predict_manual_transaction(self, fraud_system, amount: float, user_id: int, device_id: int, card_id: int, threshold: float):
        """
        Cria um nó temporário de transação, conecta aos nós selecionados e executa a inferência no modelo GNN.
        Retorna tuple: (is_fraud: bool, prob: float, new_tx_index: int)

        Observação: se não houver um transformador de features disponível no sistema,
        usamos a média das features de transação como vetor base (modelo ainda é utilizado, sem aleatoriedade).
        """
        import torch
        import copy

        if fraud_system is None or getattr(fraud_system, 'model', None) is None:
            raise RuntimeError("Modelo não disponível.")

        model = fraud_system.model
        data = getattr(fraud_system, 'train_graph', None)
        if data is None:
            raise RuntimeError("Grafo de treino não disponível.")

        # Deep copy para não alterar o grafo em sessão
        hd = copy.deepcopy(data)

        # Detectar device do modelo
        device = next(model.parameters()).device

        # Utilitários de device
        def _to_device_hetero(hdata, dev):
            # Move x_dict e edge_index_dict para o device
            for nt in hdata.node_types:
                if hasattr(hdata[nt], 'x') and hdata[nt].x is not None:
                    hdata[nt].x = hdata[nt].x.to(dev)
                if hasattr(hdata[nt], 'y') and hdata[nt].y is not None:
                    hdata[nt].y = hdata[nt].y.to(dev)
            for et in hdata.edge_types:
                if hasattr(hdata[et], 'edge_index') and hdata[et].edge_index is not None:
                    hdata[et].edge_index = hdata[et].edge_index.to(dev)
            return hdata

        hd = _to_device_hetero(hd, device)

        # Preparar nova feature de transação
        if not hasattr(hd['transaction'], 'x') or hd['transaction'].x is None:
            raise RuntimeError("Features de transação indisponíveis em hd['transaction'].x")

        tx_x = hd['transaction'].x  # [N_tx, F]
        num_tx, feat_dim = tx_x.shape

        # Tentar usar um transformador de features, se existir no sistema
        x_new = None
        feature_builder = getattr(fraud_system, 'build_transaction_features', None)
        if callable(feature_builder):
            # Ideal: o sistema sabe como transformar amount/user/device/card em vetor de features F
            x_new_np = feature_builder(amount=amount, user_id=user_id, device_id=device_id, card_id=card_id)
            x_new = torch.as_tensor(x_new_np, dtype=tx_x.dtype, device=device)
            if x_new.ndim == 1:
                x_new = x_new.view(1, -1)
        else:
            # Sem builder explícito: usar a média das features para manter consistência dimensional.
            # Observação: o modelo continua sendo usado; não há aleatoriedade na probabilidade.
            x_new = tx_x.mean(dim=0, keepdim=True)

        if x_new.shape[1] != feat_dim:
            raise RuntimeError(f"Dimensão da feature gerada ({x_new.shape[1]}) difere do esperado ({feat_dim}).")

        # Concatenar novo nó de transação
        hd['transaction'].x = torch.cat([tx_x, x_new], dim=0)
        new_tx_idx = num_tx  # índice do novo nó

        # Adicionar arestas para conectar o novo nó
        def _append_edge(hdata, edge_type, src_idx, dst_idx):
            ei = hdata[edge_type].edge_index  # [2, E]
            new_e = torch.tensor([[src_idx], [dst_idx]], dtype=ei.dtype, device=ei.device)
            hdata[edge_type].edge_index = torch.cat([ei, new_e], dim=1)

        # Encontrar tipos de aresta compatíveis no grafo
        et_user_tx = None
        et_tx_card = None
        et_tx_device = None
        for et in hd.edge_types:
            src, rel, dst = et
            if src == 'user' and dst == 'transaction':
                et_user_tx = et
            elif src == 'transaction' and dst == 'card':
                et_tx_card = et
            elif src == 'transaction' and dst == 'device':
                et_tx_device = et

        if et_user_tx is None or et_tx_card is None or et_tx_device is None:
            raise RuntimeError("Tipos de aresta necessários não encontrados no grafo: (user->transaction), (transaction->card), (transaction->device).")

        # Validar IDs fornecidos com os limites do grafo
        def _check_id(ntype, idx):
            n = int(getattr(hd[ntype], 'num_nodes', 0))
            if n == 0 and hasattr(hd[ntype], 'x') and hd[ntype].x is not None:
                n = int(hd[ntype].x.shape[0])
            if idx < 0 or idx >= n:
                raise RuntimeError(f"ID inválido para {ntype}: {idx} (esperado entre 0 e {n-1}).")

        _check_id('user', user_id)
        _check_id('card', card_id)
        _check_id('device', device_id)

        # Conectar novo nó às entidades escolhidas
        _append_edge(hd, et_user_tx, user_id, new_tx_idx)
        _append_edge(hd, et_tx_card, new_tx_idx, card_id)
        _append_edge(hd, et_tx_device, new_tx_idx, device_id)

        # Inferência
        model.eval()
        with torch.no_grad():
            logits = model(hd.x_dict, hd.edge_index_dict)  # esperado: [N_tx+1, num_classes] para 'transaction'
            probs = torch.softmax(logits, dim=1)[:, 1]
            p_new = float(probs[new_tx_idx].item())
            is_fraud = bool(p_new > threshold)

        return is_fraud, p_new, new_tx_idx

    def _build_manual_tx_figure(self, user_id: int, device_id: int, card_id: int):
        """
        Constrói uma visualização simples do 'novo nó de transação' conectado
        aos nós selecionados (user/card/device). Destaque em amarelo para o novo nó.
        """
    # numpy não é usado diretamente aqui

        G = nx.Graph()
        # Nós existentes (IDs do grafo)
        G.add_node(f"u_{user_id}", tipo='user')
        G.add_node(f"d_{device_id}", tipo='device')
        G.add_node(f"c_{card_id}", tipo='card')
        # Novo nó de transação (não existe no grafo original)
        G.add_node("t_new", tipo='transaction_new')

        # Conexões do novo nó
        G.add_edge(f"u_{user_id}", "t_new")
        G.add_edge("t_new", f"c_{card_id}")
        G.add_edge("t_new", f"d_{device_id}")

        pos = nx.spring_layout(G, seed=11)

        tipo_cores = {
            'transaction_new': '#FFD700',  # amarelo, destaque
            'user': '#2ca02c',
            'card': '#9467bd',
            'device': '#ff7f0e'
        }

        # Edges
        xe, ye = [], []
        for (u, v) in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xe += [x0, x1, None]
            ye += [y0, y1, None]
        edge_trace = go.Scatter(x=xe, y=ye, mode='lines', line=dict(color='#aaa', width=1), hoverinfo='none')

        # Nodes
        def make_node_trace(nodes_ids):
            xs, ys, texts, colors = [], [], [], []
            for n in nodes_ids:
                x, y0 = pos[n]
                xs.append(x)
                ys.append(y0)
                attrs = G.nodes[n]
                tipo = attrs.get('tipo')
                colors.append(tipo_cores[tipo])
                if tipo == 'transaction_new':
                    texts.append("Nova Transação (t_new)")
                else:
                    prefixo = {'user':'Usuário','card':'Cartão','device':'Dispositivo'}[tipo]
                    texts.append(f"{prefixo} {n.split('_')[1]}")
            return go.Scatter(x=xs, y=ys, mode='markers', text=texts, hoverinfo='text',
                               marker=dict(size=12, color=colors, line=dict(width=1, color='#333')))

        nodes_u = [n for n, a in G.nodes(data=True) if a.get('tipo')=='user']
        nodes_c = [n for n, a in G.nodes(data=True) if a.get('tipo')=='card']
        nodes_d = [n for n, a in G.nodes(data=True) if a.get('tipo')=='device']
        nodes_tn = [n for n, a in G.nodes(data=True) if a.get('tipo')=='transaction_new']

        fig = go.Figure()
        fig.add_trace(edge_trace)
        if nodes_u:
            fig.add_trace(make_node_trace(nodes_u).update(name='Usuário'))
        if nodes_c:
            fig.add_trace(make_node_trace(nodes_c).update(name='Cartão'))
        if nodes_d:
            fig.add_trace(make_node_trace(nodes_d).update(name='Dispositivo'))
        if nodes_tn:
            fig.add_trace(make_node_trace(nodes_tn).update(name='Nova Transação'))

        fig.update_layout(
            title='Nova Transação Conectada (nó adicionado em destaque)',
            showlegend=True,
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    def _compute_user_fraud_metrics(self, fraud_system, threshold: float) -> pd.DataFrame | None:
        """Calcula métricas agregadas por usuário a partir do grafo e do modelo.
        Retorna DataFrame com colunas: user_id, total_tx, fraud_count, fraud_rate, pred_count, pred_rate, avg_prob.
        """
        try:
            import torch
            if fraud_system is None or getattr(fraud_system, 'model', None) is None:
                return None
            model = fraud_system.model
            data = getattr(fraud_system, 'train_graph', None)
            if data is None:
                return None

            # Encontrar aresta user->transaction
            et_user_tx = None
            for et in data.edge_types:
                src, rel, dst = et
                if src == 'user' and dst == 'transaction':
                    et_user_tx = et
                    break
            if et_user_tx is None:
                return None

            ei = data[et_user_tx].edge_index
            u_idx = ei[0].cpu().numpy()
            t_idx = ei[1].cpu().numpy()

            # Remover duplicados (mesmo user-tx múltiplas arestas)
            map_df = pd.DataFrame({'user_id': u_idx, 'tx_id': t_idx}).drop_duplicates()

            # Labels verdadeiros (se existirem)
            y_true = None
            if hasattr(data['transaction'], 'y') and data['transaction'].y is not None:
                y_true = data['transaction'].y.detach().cpu().numpy()

            # Predições do modelo (probabilidade por transação)
            model.eval()
            with torch.no_grad():
                logits = model(data.x_dict, data.edge_index_dict)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            # Agregar por usuário
            def _agg_group(grp):
                tx_ids = grp['tx_id'].to_numpy()
                total = len(tx_ids)
                fraud_cnt = int(y_true[tx_ids].sum()) if y_true is not None else np.nan
                fraud_rate = (fraud_cnt / total) if y_true is not None and total > 0 else np.nan
                pred_cnt = int((probs[tx_ids] > threshold).sum())
                pred_rate = pred_cnt / total if total > 0 else 0.0
                avg_prob = float(probs[tx_ids].mean()) if total > 0 else 0.0
                return pd.Series({
                    'total_tx': int(total),
                    'fraud_count': fraud_cnt,
                    'fraud_rate': fraud_rate,
                    'pred_count': int(pred_cnt),
                    'pred_rate': float(pred_rate),
                    'avg_prob': avg_prob,
                })

            res = map_df.groupby('user_id', as_index=False).apply(_agg_group).reset_index(drop=True)
            # Garantir tipos numéricos
            res['user_id'] = res['user_id'].astype(int)
            return res
        except Exception:
            return None

    def run(self):
        """Executa o dashboard."""
        # Inicializar chaves padrão do session_state para evitar AttributeError
        for k, v in {
            'model_loaded': False,
            'model_trained': False,
            'training_in_progress': False,
            'current_threshold': 0.5,
            'history': []
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v
        # Renderizar sidebar e obter página selecionada
        page, threshold = self.render_sidebar()
        
        # Atualizar threshold no session state se mudou
        if 'current_threshold' not in st.session_state or st.session_state.current_threshold != threshold:
            st.session_state.current_threshold = threshold
        
        # Verificar se modelo foi treinado
        model_trained = self.is_model_trained()
        
        # Renderizar página apropriada
        if page == " Configurações":
            self.render_settings_page()
        elif not model_trained:
            # Se não há modelo treinado, mostrar página vazia
            if page == " Overview":
                self.render_empty_page("Overview", "")
            elif page == " Métricas":
                self.render_empty_page("Métricas Detalhadas", "")
            elif page == " Análise de Grafos":
                self.render_empty_page("Análise de Grafos", "")
            elif page == " Detecção em Tempo Real":
                self.render_empty_page("Detecção em Tempo Real", "")
            elif page == " Histórico":
                self.render_empty_page("Histórico de Transações", "")
        else:
            # Modelo treinado, renderizar páginas funcionais
            if page == " Overview":
                self.render_overview_page()
            elif page == " Métricas":
                self.render_metrics_page(threshold)
            elif page == " Análise de Grafos":
                self.render_graph_analysis_page()
            elif page == " Interpretabilidade":
                self.render_explainability_page()
            elif page == " Detecção em Tempo Real":
                self.render_realtime_page(threshold)
            elif page == " Histórico":
                self.render_history_page()

    def render_explainability_page(self):
        """Página de interpretabilidade: explica uma predição para um nó de transação."""
        if not self.is_model_trained():
            self.render_empty_page("Interpretabilidade", "")
            return

        st.title(" Interpretabilidade de Predições")
        fraud_system = st.session_state.get('fraud_system')
        if not fraud_system:
            st.error(" Sistema de fraude não disponível")
            return

        colsel1, colsel2, colsel3 = st.columns([2,2,1])
        # Detectar última transação manual a partir do session_state ou histórico
        manual_ctx = st.session_state.get('last_manual_ctx')
        if manual_ctx is None:
            _hist = st.session_state.get('history') or []
            _manual_entries = [h for h in _hist if h.get('method') == 'manual_gnn']
            manual_ctx = _manual_entries[0] if _manual_entries else None
        use_last = manual_ctx is not None
        with colsel1:
            source = st.selectbox(
                "Escolha a origem",
                options=(['Última transação manual'] if use_last else []) + ['Selecionar por índice'],
                index=0 if use_last else 0
            )
        idx = None
        if source == 'Última transação manual' and use_last:
            st.info("Usando a última transação manual (o nó é reconstruído temporariamente para explicação)")
        elif not use_last:
            st.caption("Dica: crie uma transação manual na aba 'Detecção em Tempo Real' para habilitar esta opção.")
        else:
            data = getattr(fraud_system, 'train_graph', None)
            if data is None:
                st.warning("Grafo indisponível.")
                return
            if hasattr(data, 'x_dict'):
                total = int(data['transaction'].num_nodes)
            else:
                total = int(data.num_nodes)
            with colsel2:
                idx = st.number_input("Índice da transação", min_value=0, max_value=max(0, total-1), value=0, step=1)

        st.markdown("---")
        st.subheader("Configurações da explicação")
        e1, e2, e3, e4 = st.columns([2,2,2,2])
        with e1:
            baseline = st.selectbox(
                "Baseline para oclusão",
                options=["zero", "mean", "median", "sampled"],
                index=1
            )
        with e2:
            n_samples = st.slider("Amostras (se sampled)", min_value=1, max_value=20, value=5, step=1)
        with e3:
            positive_only = st.checkbox("Somente deltas positivos", value=True,
                                        help="Quando marcado, deltas negativos são truncados em 0.")
        with e4:
            normalize = st.checkbox("Normalizar importâncias (soma=1)", value=False)

        kcol1, kcol2 = st.columns([2,2])
        with kcol1:
            top_k = st.slider("Top-K features para exibir", min_value=5, max_value=100, value=20, step=5)
        with kcol2:
            show_rel = st.checkbox("Exibir importância por relação", value=True)

        explain = st.button(" Explicar predição", type="primary")
        if explain and ((source == 'Última transação manual' and use_last) or (idx is not None)):
            with st.spinner("Calculando explicação por oclusão (features e relações)..."):
                try:
                    if source == 'Última transação manual' and manual_ctx is not None:
                        exp = self._explain_manual_transaction_occlusion(
                            fraud_system,
                            amount=float(manual_ctx.get('amount', 0.0)),
                            user_id=int(manual_ctx.get('user_id', 0)),
                            device_id=int(manual_ctx.get('device_id', 0)),
                            card_id=int(manual_ctx.get('card_id', 0)),
                            baseline=baseline,
                            n_samples=int(n_samples),
                            positive_only=bool(positive_only)
                        )
                    else:
                        exp = self._explain_transaction_occlusion(
                            fraud_system,
                            int(idx),
                            baseline=baseline,
                            n_samples=int(n_samples),
                            positive_only=bool(positive_only)
                        )
                except Exception as e:
                    exp = None
                    st.error(f"Falha na explicação: {e}")

            if exp is not None:
                feat_names = exp.get('feature_names')
                feat_importance = exp.get('feature_importance')
                rel_importance = exp.get('relation_importance')
                base_prob = exp.get('base_prob')

                st.subheader("Resultado da Predição")
                st.write(f"Probabilidade base (classe fraude): {base_prob:.4f}")

                if feat_importance is not None and len(feat_importance) > 0:
                    st.subheader("Importância das Features (oclusão)")
                    vals = np.array(feat_importance, dtype=float)
                    if normalize:
                        s = np.sum(np.abs(vals)) if not positive_only else np.sum(vals)
                        if s > 0:
                            vals = vals / s
                    k = min(int(top_k), len(vals))
                    order = np.argsort(-vals)[:k]
                    x_labels = [feat_names[i] if feat_names is not None else f"feat_{i}" for i in order]
                    y_vals = [float(vals[i]) for i in order]
                    figf = go.Figure(go.Bar(x=x_labels, y=y_vals, marker_color="#1f77b4"))
                    figf.update_layout(height=350, xaxis_title="Feature", yaxis_title="Δ prob (base - ocluída)", xaxis_tickangle=45)
                    st.plotly_chart(figf, use_container_width=True)

                    # Exportação CSV
                    try:
                        import pandas as _pd
                        df_imp = _pd.DataFrame({
                            'feature': feat_names if feat_names is not None else [f"feat_{i}" for i in range(len(feat_importance))],
                            'importance': feat_importance
                        })
                        if normalize:
                            df_imp['normalized_importance'] = (df_imp['importance'] / (df_imp['importance'].sum() if positive_only else df_imp['importance'].abs().sum())).fillna(0.0)
                        csv = df_imp.to_csv(index=False).encode('utf-8')
                        st.download_button("Baixar importâncias de features (CSV)", data=csv, file_name=f"feature_importances_tx_{idx}.csv", mime="text/csv")
                    except Exception:
                        pass

                if show_rel and rel_importance:
                    st.subheader("Importância por Tipo de Relação")
                    labels = list(rel_importance.keys())
                    vals = [float(rel_importance[k]) for k in labels]
                    figr = go.Figure(go.Bar(x=labels, y=vals, marker_color="#d62728"))
                    figr.update_layout(height=300, xaxis_title="Relação removida", yaxis_title="Δ prob (base - sem relação)")
                    st.plotly_chart(figr, use_container_width=True)
                    # CSV relações
                    try:
                        import pandas as _pd
                        df_rel = _pd.DataFrame({'relation': labels, 'importance': vals})
                        csv2 = df_rel.to_csv(index=False).encode('utf-8')
                        st.download_button("Baixar importâncias de relações (CSV)", data=csv2, file_name=f"relation_importances_tx_{idx}.csv", mime="text/csv")
                    except Exception:
                        pass

    def _explain_transaction_occlusion(self, fraud_system, tx_index: int, baseline: str = "mean", n_samples: int = 5, positive_only: bool = True):
        """Explica predição para um nó de transação via oclusão de features e remoção de relações.
        Retorna dict com: base_prob, feature_importance (np.array), feature_names (list), relation_importance (dict).
        """
        import torch
        import copy
        if fraud_system is None or getattr(fraud_system, 'model', None) is None:
            raise RuntimeError("Modelo não disponível")
        model = fraud_system.model
        data = getattr(fraud_system, 'train_graph', None)
        if data is None:
            raise RuntimeError("Grafo indisponível")

        dev = next(model.parameters()).device
        hd = copy.deepcopy(data)

        # Move para device
        for nt in hd.node_types:
            if hasattr(hd[nt], 'x') and hd[nt].x is not None:
                hd[nt].x = hd[nt].x.to(dev)
            if hasattr(hd[nt], 'y') and hd[nt].y is not None:
                hd[nt].y = hd[nt].y.to(dev)
        for et in hd.edge_types:
            if hasattr(hd[et], 'edge_index') and hd[et].edge_index is not None:
                hd[et].edge_index = hd[et].edge_index.to(dev)

        # Prob base
        model.eval()
        with torch.no_grad():
            logits = model(hd.x_dict, hd.edge_index_dict)
            base_prob = torch.softmax(logits[tx_index], dim=0)[1].item()

        # Importância por feature (oclusão com baseline configurável)
        x_tx = hd['transaction'].x
        feat_dim = int(x_tx.shape[1])
        feat_names = [f"feat_{i}" for i in range(feat_dim)]
        deltas = np.zeros(feat_dim, dtype=float)
        original = x_tx[tx_index].clone()
        # Pré-computar estatísticas para baselines
        col_means = x_tx.mean(dim=0)
        col_medians = x_tx.median(dim=0).values
        # Usar amostras do próprio conjunto como baseline se 'sampled'
        for d in range(feat_dim):
            saved = float(original[d].item())
            # Escolher o valor de referência
            if baseline == 'zero':
                ref_vals = [0.0]
            elif baseline == 'median':
                ref_vals = [float(col_medians[d].item())]
            elif baseline == 'sampled':
                # Amostrar n_samples valores dessa coluna
                vals_col = x_tx[:, d]
                if vals_col.numel() > 0:
                    # torch.randint para índices
                    idxs = torch.randint(0, vals_col.shape[0], (max(1, int(n_samples)),), device=vals_col.device)
                    ref_vals = [float(vals_col[i].item()) for i in idxs]
                else:
                    ref_vals = [float(col_means[d].item())]
            else:  # 'mean' default
                ref_vals = [float(col_means[d].item())]

            # Agregar sobre amostras (se houver mais de uma)
            prob2_vals = []
            for rv in ref_vals:
                x_tx[tx_index, d] = rv
                with torch.no_grad():
                    logits2 = model(hd.x_dict, hd.edge_index_dict)
                    prob2 = torch.softmax(logits2[tx_index], dim=0)[1].item()
                prob2_vals.append(prob2)
            prob2_mean = float(np.mean(prob2_vals))
            delta = base_prob - prob2_mean
            if positive_only:
                delta = max(0.0, delta)
            deltas[d] = float(delta)
            x_tx[tx_index, d] = saved

        # Importância por relação: remover conexões do nó para cada tipo incidente
        rel_importance = {}
        for et in hd.edge_types:
            s, r, t = et
            ei = hd[et].edge_index
            # Arestas que chegam ao nó de transação
            if t == 'transaction':
                mask_inc = (ei[1] == tx_index)
                if mask_inc.any().item():
                    ei_saved = ei.clone()
                    hd[et].edge_index = ei[:, ~mask_inc]
                    with torch.no_grad():
                        logits3 = model(hd.x_dict, hd.edge_index_dict)
                        prob3 = torch.softmax(logits3[tx_index], dim=0)[1].item()
                    delta = base_prob - prob3
                    if positive_only:
                        delta = max(0.0, delta)
                    rel_importance[f"{s}→transaction ({r})"] = float(delta)
                    hd[et].edge_index = ei_saved
            # Arestas que saem do nó de transação
            if s == 'transaction':
                mask_out = (ei[0] == tx_index)
                if mask_out.any().item():
                    ei_saved = ei.clone()
                    hd[et].edge_index = ei[:, ~mask_out]
                    with torch.no_grad():
                        logits3 = model(hd.x_dict, hd.edge_index_dict)
                        prob3 = torch.softmax(logits3[tx_index], dim=0)[1].item()
                    delta = base_prob - prob3
                    if positive_only:
                        delta = max(0.0, delta)
                    rel_importance[f"transaction→{t} ({r})"] = float(delta)
                    hd[et].edge_index = ei_saved

        return {
            'base_prob': float(base_prob),
            'feature_importance': deltas,
            'feature_names': feat_names,
            'relation_importance': rel_importance,
        }

    def _explain_manual_transaction_occlusion(self, fraud_system, amount: float, user_id: int, device_id: int, card_id: int,
                                              baseline: str = "mean", n_samples: int = 5, positive_only: bool = True):
        """Reconstrói a transação manual (como em _predict_manual_transaction), obtém o índice do novo nó
        e delega para _explain_transaction_occlusion usando o índice dessa transação adicionada temporariamente.
        """
        import torch
        import copy

        if fraud_system is None or getattr(fraud_system, 'model', None) is None:
            raise RuntimeError("Modelo não disponível.")

        model = fraud_system.model
        data = getattr(fraud_system, 'train_graph', None)
        if data is None:
            raise RuntimeError("Grafo indisponível.")

        dev = next(model.parameters()).device
        hd = copy.deepcopy(data)

        # mover para device
        for nt in hd.node_types:
            if hasattr(hd[nt], 'x') and hd[nt].x is not None:
                hd[nt].x = hd[nt].x.to(dev)
        for et in hd.edge_types:
            if hasattr(hd[et], 'edge_index') and hd[et].edge_index is not None:
                hd[et].edge_index = hd[et].edge_index.to(dev)

        # Criar a nova transação (similar a _predict_manual_transaction)
        tx_x = hd['transaction'].x
        num_tx, _ = tx_x.shape
        # baseline de features da nova transação: usar média (como default) para não introduzir variáveis externas
        x_new = tx_x.mean(dim=0, keepdim=True)
        hd['transaction'].x = torch.cat([tx_x, x_new], dim=0)
        new_tx_idx = num_tx

        # localizar tipos de aresta
        et_user_tx = None
        et_tx_card = None
        et_tx_device = None
        for et in hd.edge_types:
            s, r, t = et
            if s == 'user' and t == 'transaction':
                et_user_tx = et
            elif s == 'transaction' and t == 'card':
                et_tx_card = et
            elif s == 'transaction' and t == 'device':
                et_tx_device = et
        if et_user_tx is None or et_tx_card is None or et_tx_device is None:
            raise RuntimeError("Tipos de aresta necessários não encontrados no grafo.")

        def _append_edge(hdata, edge_type, src_idx, dst_idx):
            ei = hdata[edge_type].edge_index
            new_e = torch.tensor([[src_idx], [dst_idx]], dtype=ei.dtype, device=ei.device)
            hdata[edge_type].edge_index = torch.cat([ei, new_e], dim=1)

        # validar ids simples (limitados pelo número de nós)
        def _cap(ntype, idx):
            n = int(getattr(hd[ntype], 'num_nodes', 0))
            if n == 0 and hasattr(hd[ntype], 'x') and hd[ntype].x is not None:
                n = int(hd[ntype].x.shape[0])
            return max(0, min(int(idx), n - 1 if n > 0 else 0))

        user_id = _cap('user', user_id)
        card_id = _cap('card', card_id)
        device_id = _cap('device', device_id)

        _append_edge(hd, et_user_tx, user_id, new_tx_idx)
        _append_edge(hd, et_tx_card, new_tx_idx, card_id)
        _append_edge(hd, et_tx_device, new_tx_idx, device_id)

        # Agora explicar o novo índice; para isso, tiramos proveito do mesmo método, mas sobre 'hd'.
        # Para reusar a lógica sem duplicar, criamos um objeto leve com os atributos necessários.
        from types import SimpleNamespace
        tmp = SimpleNamespace(model=model, train_graph=hd)

        return self._explain_transaction_occlusion(
            tmp, int(new_tx_idx), baseline=baseline, n_samples=n_samples, positive_only=positive_only
        )


def main():
    """Função principal do dashboard."""
    dashboard = FraudDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
