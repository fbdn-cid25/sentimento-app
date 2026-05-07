"""
app.py — Dashboard Interativo de Análise de Incidentes de TI.

Este dashboard utiliza Streamlit e Plotly para análise descritiva e preditiva
de incidentes de TI, usando um modelo treinado via DagsHub/MLflow.

Dependências Adicionais Requeridas:
    pip install plotly

Execute com:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ── 1. Configuração da página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard de Incidentes de TI",
    page_icon="🛠️",
    layout="wide",
)

def inject_custom_css():
    st.markdown("""
        <style>
        /* Cabeçalho Vermelho Corporativo */
        .corporate-header {
            background-color: #D32F2F;
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .corporate-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }
        .corporate-header p {
            color: #FFEBEE !important;
            margin: 5px 0 0 0;
            font-size: 1.1rem;
            font-weight: 400;
        }
        
        /* Ajuste de Métricas */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #D32F2F !important;
            font-weight: bold;
        }
        
        /* Container styling */
        .block-container {
            padding-top: 2rem !important;
        }
        
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ── 2. Funções de Cache (Utilitários) ────────────────────────────────────────

@st.cache_resource
def load_model():
    """
    Carrega o modelo salvo uma única vez (singleton).
    O modelo esperado é um Pipeline Scikit-Learn (TF-IDF + Classificador).
    """
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Certifique-se de que 'model.pkl' existe. Detalhes: {e}")
        return None

@st.cache_data
def load_data():
    """
    Carrega o dataset histórico de incidentes para a Visão Geral.
    Resolve problemas de encoding e delimitadores.
    """
    try:
        # Tenta ler com utf-8 primeiro, usando sep=";"
        df = pd.read_csv("data/incidente.csv", sep=";", encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Instancia o modelo e dados
pipeline = load_model()
df_historico = load_data()

# Paleta de Cores Corporativa para Gráficos
PLOT_COLORS = ['#D32F2F', '#1976D2', '#388E3C', '#FBC02D', '#7B1FA2', '#0097A7', '#E64A19', '#5D4037']

# ── 3. Navegação Lateral ─────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/settings--v1.png", width=64)
    st.title("Navegação")
    st.markdown("Selecione a ferramenta de análise abaixo:")
    
    view = st.radio(
        "",
        ["📊 Visão Geral", 
         "🔍 Analisador Individual", 
         "📂 Predição em Lote", 
         "📈 Avaliação de Modelo"]
    )
    
    st.divider()
    st.info(
        "**Modelo Operacional:**\n\n"
        "TF-IDF + Logistic Regression\n"
        "(Treinado via MLflow/DagsHub)"
    )
    st.caption("v1.0.0 | © 2026 IT Operations")

# ── 4. Lógica de Roteamento de Views ─────────────────────────────────────────

# -----------------------------------------------------------------------------
# VIEW 1: VISÃO GERAL (OVERVIEW)
# -----------------------------------------------------------------------------
if view == "📊 Visão Geral":
    st.markdown("""
        <div class="corporate-header">
            <h1>📊 Visão Geral dos Incidentes</h1>
            <p>Análise consolidada e distribuição do histórico de chamados de TI</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df_historico.empty:
        st.warning("Não foi possível carregar o dataset histórico.")
    else:
        # Filtro textual (Sidebar)
        busca = st.sidebar.text_input("🔍 Filtrar incidentes por texto:")
        if busca:
            df_view = df_historico[df_historico['detalhe_incidente'].str.contains(busca, case=False, na=False)]
        else:
            df_view = df_historico
            
        total_regs = len(df_view)
        missing_labels = df_view['tipo_incidente'].isnull().sum()
        
        if total_regs > 0 and not df_view['tipo_incidente'].dropna().empty:
            val_counts = df_view['tipo_incidente'].value_counts()
            top_classe = val_counts.index[0]
            top_pct = (val_counts.iloc[0] / val_counts.sum()) * 100
        else:
            top_classe = "N/A"
            top_pct = 0.0

        st.markdown("### 📌 Indicadores Principais")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Incidentes", f"{total_regs:,}".replace(",", "."))
        with col2:
            st.metric("Categorias Únicas", df_view['tipo_incidente'].nunique())
        with col3:
            st.metric("Classe Frequente", top_classe, f"{top_pct:.1f}% do total", delta_color="off")
        with col4:
            st.metric("Registros S/ Rótulo", missing_labels)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not df_view.empty:
            col_chart1, col_chart2 = st.columns([1.5, 1])
            
            dist_classes = df_view['tipo_incidente'].value_counts().reset_index()
            dist_classes.columns = ['Tipo de Incidente', 'Quantidade']
            
            with col_chart1:
                st.markdown("#### Distribuição por Categoria")
                fig_bar = px.bar(
                    dist_classes, 
                    x='Quantidade', 
                    y='Tipo de Incidente', 
                    orientation='h',
                    text='Quantidade',
                    color='Quantidade',
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### Proporção Principal")
                top_5 = dist_classes.head(5)
                outros_qtde = dist_classes.iloc[5:]['Quantidade'].sum()
                if outros_qtde > 0:
                    top_5 = pd.concat([top_5, pd.DataFrame([{'Tipo de Incidente': 'Outros', 'Quantidade': outros_qtde}])])
                    
                fig_pie = px.pie(
                    top_5, 
                    values='Quantidade', 
                    names='Tipo de Incidente', 
                    hole=0.45,
                    color_discrete_sequence=PLOT_COLORS
                )
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.divider()
            st.markdown("### 📋 Amostra de Dados")
            st.dataframe(df_view.head(100), use_container_width=True, height=400)

# -----------------------------------------------------------------------------
# VIEW 2: PREDICAÇÃO INDIVIDUAL
# -----------------------------------------------------------------------------
elif view == "🔍 Analisador Individual":
    st.markdown("""
        <div class="corporate-header">
            <h1>🔍 Analisador Individual de Chamados</h1>
            <p>Insira os detalhes de um incidente para classificação inteligente via IA</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not pipeline:
        st.error("O modelo não está disponível no momento.")
        st.stop()
        
    with st.container(border=True):
        st.markdown("#### Descreva o incidente:")
        texto = st.text_area(
            "", 
            height=150, 
            label_visibility="collapsed",
            placeholder="Exemplo: O usuário da contabilidade relatou que o sistema ERP não conecta e apresenta erro de timeout..."
        )
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            submit = st.button("🚀 Analisar Incidente", type="primary", use_container_width=True)
        
    if submit:
        if not texto or len(texto.strip()) < 5:
            st.warning("Por favor, insira um texto com pelo menos 5 caracteres para uma análise consistente.")
        else:
            with st.spinner("Processando através do modelo de Machine Learning..."):
                texto_limpo = texto.lower().strip()
                try:
                    pred = pipeline.predict([texto_limpo])[0]
                    probas = pipeline.predict_proba([texto_limpo])[0]
                    classes = pipeline.classes_
                    
                    st.success("Análise concluída com sucesso!")
                    
                    col_result1, col_result2 = st.columns([1, 1])
                    
                    with col_result1:
                        st.markdown("### 🎯 Categoria Sugerida")
                        st.info(f"**{pred}**", icon="📌")
                        
                    with col_result2:
                        st.markdown("### 📊 Nível de Confiança (Top 3)")
                        prob_df = pd.DataFrame({"Classe": classes, "Probabilidade": probas})
                        prob_df = prob_df.sort_values(by="Probabilidade", ascending=False).head(3)
                        
                        for _, row in prob_df.iterrows():
                            pct = float(row['Probabilidade'])
                            st.progress(pct, text=f"{row['Classe']} ({pct:.1%})")
                            
                except Exception as e:
                    st.error(f"Erro durante a predição. Detalhes do sistema: {e}")

# -----------------------------------------------------------------------------
# VIEW 3: PREDICAÇÃO EM LOTE
# -----------------------------------------------------------------------------
elif view == "📂 Predição em Lote":
    st.markdown("""
        <div class="corporate-header">
            <h1>📂 Classificação em Lote</h1>
            <p>Automatize a triagem de múltiplos chamados enviando um arquivo CSV</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not pipeline:
        st.error("O modelo não está disponível.")
        st.stop()
        
    with st.expander("ℹ️ Instruções para Upload", expanded=True):
        st.write("""
        1. O arquivo deve estar no formato **.CSV**.
        2. Certifique-se de que o arquivo possua uma coluna chamada **`detalhe_incidente`** ou **`texto`**.
        3. O processamento será feito automaticamente após a submissão.
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Arraste ou selecione o arquivo CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo. Verifique a formatação. ({e})")
            st.stop()
            
        col_texto = "detalhe_incidente" if "detalhe_incidente" in df_upload.columns else ("texto" if "texto" in df_upload.columns else None)
            
        if not col_texto:
            st.error("Erro: Coluna de texto não localizada. O CSV deve conter 'detalhe_incidente' ou 'texto'.")
        else:
            if st.button("Executar Triagem em Lote", type="primary"):
                with st.spinner(f"Classificando {len(df_upload)} registros..."):
                    df_upload[col_texto] = df_upload[col_texto].fillna("").astype(str)
                    textos = df_upload[col_texto].str.lower().str.strip()
                    
                    preds = pipeline.predict(textos)
                    probas = pipeline.predict_proba(textos)
                    
                    df_upload["predito_tipo_incidente"] = preds
                    df_upload["confianca_maxima"] = probas.max(axis=1)
                    
                    if "tipo_incidente" in df_upload.columns:
                        df_upload["acertou"] = df_upload["tipo_incidente"] == df_upload["predito_tipo_incidente"]
                        
                    st.success("✅ Classificação concluída!")
                    
                    st.markdown("### 📄 Pré-visualização dos Resultados")
                    st.dataframe(df_upload.head(50), use_container_width=True)
                    
                    csv_bytes = df_upload.to_csv(index=False, sep=";").encode("utf-8-sig")
                    st.download_button(
                        label="📥 Baixar Base Classificada (CSV)",
                        data=csv_bytes,
                        file_name="incidentes_classificados.csv",
                        mime="text/csv",
                        type="primary"
                    )

# -----------------------------------------------------------------------------
# VIEW 4: AVALIAÇÃO DO MODELO
# -----------------------------------------------------------------------------
elif view == "📈 Avaliação de Modelo":
    st.markdown("""
        <div class="corporate-header">
            <h1>📈 Auditoria e Avaliação do Modelo</h1>
            <p>Valide a acurácia da IA contra uma base real (Ground Truth)</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not pipeline:
        st.error("O modelo não está disponível.")
        st.stop()
        
    uploaded_file = st.file_uploader("Selecione a base de testes (CSV)", type=["csv"], help="O arquivo deve conter as colunas 'detalhe_incidente' e 'tipo_incidente'")
    
    if uploaded_file:
        df_eval = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        if "detalhe_incidente" not in df_eval.columns or "tipo_incidente" not in df_eval.columns:
            st.error("O arquivo CSV precisa ter obrigatoriamente as colunas `detalhe_incidente` e `tipo_incidente`.")
        else:
            df_eval = df_eval.dropna(subset=["tipo_incidente"])
            if len(df_eval) == 0:
                st.error("Não há registros válidos na coluna alvo.")
                st.stop()
                
            with st.spinner("Calculando métricas de performance..."):
                textos = df_eval["detalhe_incidente"].fillna("").astype(str).str.lower().str.strip()
                y_true = df_eval["tipo_incidente"]
                y_pred = pipeline.predict(textos)
                
                acc = accuracy_score(y_true, y_pred)
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
                p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                
                st.markdown("### 🎯 Desempenho Global")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Acurácia Geral", f"{acc:.2%}")
                with c2:
                    st.metric("F1 Score (Weighted)", f"{f1_weight:.3f}")
                with c3:
                    st.metric("F1 Score (Macro)", f"{f1_macro:.3f}")
                with c4:
                    st.metric("Volume de Teste", len(y_true))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_cm, col_err = st.columns([1.2, 1])
                
                with col_cm:
                    st.markdown("#### Matriz de Confusão")
                    labels = sorted(list(set(y_true) | set(y_pred)))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    
                    fig_cm = px.imshow(
                        cm,
                        x=labels,
                        y=labels,
                        labels=dict(x="Classe Prevista", y="Classe Real", color="Ocorrências"),
                        text_auto=True,
                        color_continuous_scale='Reds',
                        aspect="auto"
                    )
                    fig_cm.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with col_err:
                    st.markdown("#### Principais Divergências")
                    df_eval["predicao"] = y_pred
                    df_erros = df_eval[df_eval["tipo_incidente"] != df_eval["predicao"]]
                    
                    if len(df_erros) > 0:
                        st.warning(f"Identificados **{len(df_erros)}** desvios de classificação.")
                        
                        df_erros["confusao"] = df_erros["tipo_incidente"] + " ➔ " + df_erros["predicao"]
                        top_confusoes = df_erros["confusao"].value_counts().reset_index()
                        top_confusoes.columns = ["Real ➔ Previsto", "Qtd"]
                        
                        st.dataframe(top_confusoes.head(10), hide_index=True, use_container_width=True)
                    else:
                        st.success("Desempenho perfeito nesta base de teste!")

                if len(df_erros) > 0:
                    with st.expander("🔍 Ver Amostra de Erros Textuais"):
                        amostra = df_erros[["detalhe_incidente", "tipo_incidente", "predicao"]].head(10)
                        st.dataframe(amostra, use_container_width=True, hide_index=True)
