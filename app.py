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

# ── 3. Navegação Lateral ─────────────────────────────────────────────────────

st.sidebar.title("Navegação")
st.sidebar.markdown("Selecione a ferramenta de análise:")
view = st.sidebar.radio(
    "",
    ["1) Visão Geral (Overview)", 
     "2) Analisador Individual", 
     "3) Predição em Lote", 
     "4) Avaliação (Ground Truth)"]
)

st.sidebar.divider()
st.sidebar.caption(
    "Modelo: TF-IDF + Logistic Regression\n\n"
    "Treinado via MLflow/DagsHub"
)

# ── 4. Lógica de Roteamento de Views ─────────────────────────────────────────

# -----------------------------------------------------------------------------
# VIEW 1: VISÃO GERAL (OVERVIEW)
# -----------------------------------------------------------------------------
if view == "1) Visão Geral (Overview)":
    st.title("📊 Visão Geral dos Incidentes")
    st.markdown("Visão consolidada do banco de dados histórico (`data/incidente.csv`).")
    
    if df_historico.empty:
        st.warning("Não foi possível carregar o dataset histórico.")
    else:
        # Preprocessamento básico de contagem
        total_regs = len(df_historico)
        num_classes = df_historico['tipo_incidente'].nunique()
        missing_labels = df_historico['tipo_incidente'].isnull().sum()
        
        # Classe mais frequente
        val_counts = df_historico['tipo_incidente'].value_counts()
        top_classe = val_counts.index[0]
        top_pct = (val_counts.iloc[0] / val_counts.sum()) * 100
        
        # Filtro textual (Sidebar)
        busca = st.sidebar.text_input("Filtrar por texto no incidente:")
        if busca:
            df_view = df_historico[df_historico['detalhe_incidente'].str.contains(busca, case=False, na=False)]
        else:
            df_view = df_historico
            
        st.subheader("Indicadores Principais")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Incidentes", f"{len(df_view):,}".replace(",", "."))
        col2.metric("Nº de Categorias", df_view['tipo_incidente'].nunique())
        col3.metric("Classe Mais Frequente", top_classe, f"{top_pct:.1f}% do total", delta_color="off")
        col4.metric("Registros Sem Rótulo (Nulos)", missing_labels)
        
        st.divider()
        st.subheader("Distribuição de Classes")
        
        col_chart1, col_chart2 = st.columns(2)
        
        dist_classes = df_view['tipo_incidente'].value_counts().reset_index()
        dist_classes.columns = ['Tipo de Incidente', 'Quantidade']
        
        with col_chart1:
            fig_bar = px.bar(
                dist_classes, 
                x='Quantidade', 
                y='Tipo de Incidente', 
                orientation='h',
                title="Contagem por Categoria",
                text='Quantidade',
                color='Quantidade',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            # Pegando as top 5 e agrupando o resto como 'Outros' para não poluir o Donut
            top_5 = dist_classes.head(5)
            outros_qtde = dist_classes.iloc[5:]['Quantidade'].sum()
            if outros_qtde > 0:
                top_5 = pd.concat([top_5, pd.DataFrame([{'Tipo de Incidente': 'Outros', 'Quantidade': outros_qtde}])])
                
            fig_pie = px.pie(
                top_5, 
                values='Quantidade', 
                names='Tipo de Incidente', 
                hole=0.4,
                title="Proporção das Principais Categorias"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.divider()
        st.subheader("Amostra dos Dados")
        st.dataframe(df_view.head(50), use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW 2: PREDICAÇÃO INDIVIDUAL
# -----------------------------------------------------------------------------
elif view == "2) Analisador Individual":
    st.title("🔍 Analisador de Detalhamento")
    st.markdown("Insira o detalhamento de um chamado ou incidente para que a IA preveja a sua categoria.")
    
    if not pipeline:
        st.error("O modelo não está disponível.")
        st.stop()
        
    texto = st.text_area(
        "Detalhes do Incidente", 
        height=150, 
        placeholder="Ex: Usuário relatou que a impressora principal do RH não está conectando na rede..."
    )
    
    if st.button("Classificar Incidente", type="primary"):
        if not texto or len(texto.strip()) < 5:
            st.error("Por favor, insira um texto com pelo menos 5 caracteres.")
        else:
            with st.spinner("Classificando..."):
                texto_limpo = texto.lower().strip()
                
                # Predição
                try:
                    pred = pipeline.predict([texto_limpo])[0]
                    probas = pipeline.predict_proba([texto_limpo])[0]
                    classes = pipeline.classes_
                    
                    st.success("Análise concluída!")
                    
                    st.markdown(f"### 🎯 Categoria Prevista: **{pred}**")
                    
                    st.divider()
                    st.subheader("Confiança do Modelo (Top 3)")
                    
                    # Ordena as probabilidades
                    prob_df = pd.DataFrame({"Classe": classes, "Probabilidade": probas})
                    prob_df = prob_df.sort_values(by="Probabilidade", ascending=False).head(3)
                    
                    for _, row in prob_df.iterrows():
                        pct = float(row['Probabilidade'])
                        st.progress(pct, text=f"**{row['Classe']}**: {pct:.1%}")
                        
                except Exception as e:
                    st.error(f"Erro ao realizar a predição. Detalhes: {e}")

# -----------------------------------------------------------------------------
# VIEW 3: PREDICAÇÃO EM LOTE
# -----------------------------------------------------------------------------
elif view == "3) Predição em Lote":
    st.title("📂 Classificação de Incidentes em Lote")
    st.markdown("Faça o upload de um arquivo CSV contendo os chamados. O sistema classificará todos automaticamente.")
    
    if not pipeline:
        st.error("O modelo não está disponível.")
        st.stop()
        
    st.info("O CSV precisa ter o delimitador correto (ponto e vírgula ou vírgula) e uma coluna chamada **`detalhe_incidente`** ou **`texto`**.")
    
    uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])
    
    if uploaded_file:
        try:
            # Tenta separar por vírgula, se der erro de coluna, tenta ponto-e-vírgula
            df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception as e:
            st.error(f"Erro ao ler o arquivo CSV: {e}")
            st.stop()
            
        col_texto = None
        if "detalhe_incidente" in df_upload.columns:
            col_texto = "detalhe_incidente"
        elif "texto" in df_upload.columns:
            col_texto = "texto"
            
        if not col_texto:
            st.error("Coluna não encontrada! O CSV precisa ter 'detalhe_incidente' ou 'texto'.")
        else:
            if st.button("Executar Predição em Lote", type="primary"):
                with st.spinner(f"Processando {len(df_upload)} registros..."):
                    # Preprocessamento (evitar falha com nulos)
                    df_upload[col_texto] = df_upload[col_texto].fillna("").astype(str)
                    textos = df_upload[col_texto].str.lower().str.strip()
                    
                    # Realiza predição
                    preds = pipeline.predict(textos)
                    probas = pipeline.predict_proba(textos)
                    
                    df_upload["predito_tipo_incidente"] = preds
                    df_upload["confianca_maxima"] = probas.max(axis=1)
                    
                    # Se Ground Truth estiver disponível, marca se acertou
                    if "tipo_incidente" in df_upload.columns:
                        df_upload["acertou"] = df_upload["tipo_incidente"] == df_upload["predito_tipo_incidente"]
                        
                    st.success("Predição finalizada!")
                    
                    st.subheader("Resultado")
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Prepara download
                    csv_bytes = df_upload.to_csv(index=False, sep=";").encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Baixar Resultado (CSV)",
                        data=csv_bytes,
                        file_name="predicao_lote.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# -----------------------------------------------------------------------------
# VIEW 4: AVALIAÇÃO DO MODELO
# -----------------------------------------------------------------------------
elif view == "4) Avaliação (Ground Truth)":
    st.title("📈 Avaliação de Performance")
    st.markdown("Submeta um CSV com as colunas **`detalhe_incidente`** e **`tipo_incidente`** para calcular as métricas reais do modelo contra esses dados novos.")
    
    if not pipeline:
        st.error("O modelo não está disponível.")
        st.stop()
        
    uploaded_file = st.file_uploader("Selecione o arquivo CSV de Avaliação", type=["csv"])
    
    if uploaded_file:
        df_eval = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Verifica requisitos
        if "detalhe_incidente" not in df_eval.columns or "tipo_incidente" not in df_eval.columns:
            st.error("O arquivo CSV precisa ter obrigatoriamente as colunas `detalhe_incidente` e `tipo_incidente`.")
        else:
            # Remove nulos na variável target
            df_eval = df_eval.dropna(subset=["tipo_incidente"])
            if len(df_eval) == 0:
                st.error("Não há registros válidos na coluna `tipo_incidente` para avaliação.")
                st.stop()
                
            with st.spinner("Gerando avaliação..."):
                textos = df_eval["detalhe_incidente"].fillna("").astype(str).str.lower().str.strip()
                y_true = df_eval["tipo_incidente"]
                y_pred = pipeline.predict(textos)
                
                # Métricas globais
                acc = accuracy_score(y_true, y_pred)
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
                p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                
                st.subheader("Métricas Gerais")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Acurácia", f"{acc:.2%}")
                c2.metric("F1 Score (Weighted)", f"{f1_weight:.3f}")
                c3.metric("F1 Score (Macro)", f"{f1_macro:.3f}")
                c4.metric("Registros Avaliados", len(y_true))
                
                st.divider()
                
                # Matriz de Confusão
                st.subheader("Matriz de Confusão")
                labels = sorted(list(set(y_true) | set(y_pred)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                
                fig_cm = px.imshow(
                    cm,
                    x=labels,
                    y=labels,
                    labels=dict(x="Predito", y="Real", color="Quantidade"),
                    text_auto=True,
                    color_continuous_scale='Blues',
                    aspect="auto"
                )
                fig_cm.update_layout(xaxis_title="Classe Prevista", yaxis_title="Classe Real (Target)")
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.divider()
                
                # Ranking de erros
                st.subheader("Análise de Erros (Ranking)")
                df_eval["predicao"] = y_pred
                df_erros = df_eval[df_eval["tipo_incidente"] != df_eval["predicao"]]
                
                if len(df_erros) > 0:
                    st.error(f"Ocorreram {len(df_erros)} erros de classificação.")
                    
                    df_erros["confusao"] = "Real: " + df_erros["tipo_incidente"] + " ➔ Previsto: " + df_erros["predicao"]
                    top_confusoes = df_erros["confusao"].value_counts().reset_index()
                    top_confusoes.columns = ["Confusão (Real ➔ Previsto)", "Frequência"]
                    
                    c_rank, c_sample = st.columns([1, 2])
                    with c_rank:
                        st.markdown("**Top 10 Erros Mais Frequentes:**")
                        st.dataframe(top_confusoes.head(10), hide_index=True)
                        
                    with c_sample:
                        st.markdown("**Amostra de Textos Classificados Incorretamente:**")
                        amostra = df_erros[["detalhe_incidente", "tipo_incidente", "predicao"]].head(10)
                        st.dataframe(amostra, use_container_width=True, hide_index=True)
                else:
                    st.success("Perfeito! Nenhuma previsão incorreta encontrada.")
