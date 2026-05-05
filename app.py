"""
app.py — Interface Streamlit para análise de sentimento de reviews.

Execute com:
    streamlit run app.py

Ou via Docker:
    docker build --build-arg DAGSHUB_USER=... -t sentimento-app .
    docker run -p 8501:8501 sentimento-app
"""

import streamlit as st
import pandas as pd
import joblib

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimento",
    page_icon="💬",
    layout="centered",
)

# ── Carrega o modelo (cache: não recarrega a cada interação) ─────────────────
@st.cache_resource
def load_model():
    """
    @st.cache_resource garante que o modelo é carregado apenas uma vez.
    Sem isso, o arquivo seria lido a cada clique do usuário.
    """
    return joblib.load("model.pkl")

pipeline = load_model()

# Mapeamento visual por sentimento
EMOJI = {"positivo": "😊", "negativo": "😞", "neutro": "😐"}
COLOR = {"positivo": "green", "negativo": "red", "neutro": "orange"}

# ── Interface principal ──────────────────────────────────────────────────────
st.title("💬 Análise de Sentimento")
st.markdown(
    "Classifica reviews de produtos em **positivo**, **negativo** ou **neutro** "
    "usando um modelo TF-IDF + Regressão Logística treinado e rastreado no **DagsHub**."
)

st.divider()

# ── Seção 1: Análise de texto único ─────────────────────────────────────────
st.subheader("Analisar um review")

texto = st.text_area(
    label="Cole o texto do review aqui:",
    height=120,
    placeholder='Ex: "O produto chegou rápido e funcionou perfeitamente!"',
)

if st.button("Analisar sentimento", type="primary", use_container_width=True):
    if texto.strip():
        # Faz a predição
        pred   = pipeline.predict([texto])[0]
        probas = pipeline.predict_proba([texto])[0]
        classes = pipeline.classes_

        emoji = EMOJI.get(pred, "")
        st.markdown(f"### {emoji} Sentimento detectado: **{pred.upper()}**")

        st.markdown("##### Probabilidade por classe")
        for cls, prob in zip(classes, probas):
            st.progress(float(prob), text=f"{cls}: {prob:.0%}")
    else:
        st.warning("Digite ou cole um texto antes de analisar.")

st.divider()

# ── Seção 2: Análise em lote via CSV ────────────────────────────────────────
st.subheader("Análise em lote (CSV)")
st.markdown(
    "Envie um arquivo CSV com uma coluna chamada **`texto`**. "
    "O modelo vai classificar cada linha e você pode baixar o resultado."
)

uploaded_file = st.file_uploader(
    "Selecione o arquivo CSV",
    type=["csv"],
    help="O arquivo deve ter uma coluna chamada 'texto'",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "texto" not in df.columns:
        st.error("O CSV precisa ter uma coluna chamada 'texto'.")
    else:
        # Classifica todas as linhas
        df["sentimento"] = pipeline.predict(df["texto"].astype(str))

        # Exibe o resultado
        st.success(f"{len(df)} reviews classificados!")
        st.dataframe(df, use_container_width=True)

        # Estatísticas rápidas
        contagem = df["sentimento"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Positivos",  contagem.get("positivo", 0))
        col2.metric("Neutros",    contagem.get("neutro", 0))
        col3.metric("Negativos",  contagem.get("negativo", 0))

        # Download do resultado
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar resultado como CSV",
            data=csv_bytes,
            file_name="resultado_sentimento.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Modelo: TF-IDF + Logistic Regression · "
    "Rastreamento: MLflow + DagsHub · "
    "Deploy: Docker + Render"
)
