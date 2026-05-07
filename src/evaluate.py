"""
evaluate.py — Baixa o melhor modelo registrado no DagsHub/MLflow e salva localmente.

Este script é executado:
  - Manualmente após identificar o melhor run no DagsHub
  - Automaticamente durante o 'docker build' (via Dockerfile)

Uso:
    python src/evaluate.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, accuracy_score, f1_score
from dotenv import load_dotenv

from src.prepare_data import load_and_split

# ── Credenciais ──────────────────────────────────────────────────────────────
# Durante o docker build, as credenciais chegam como variáveis de ambiente
# (ARG no Dockerfile). Em desenvolvimento local, vêm do arquivo .env.
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# ── Configuração do MLflow ───────────────────────────────────────────────────
mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def main():
    # URI para o modelo mais recente no registro
    # "latest" sempre pega a última versão promovida
    model_uri = "models:/IT_Incident_Classifier@production"

    print(f"Baixando modelo: {model_uri}")
    print(f"Fonte: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")

    # Carrega o modelo diretamente do servidor MLflow do DagsHub
    pipeline = mlflow.sklearn.load_model(model_uri)
    print("Modelo carregado com sucesso!")

    # Avalia no conjunto de teste para confirmar que está funcionando
    _, X_test, _, y_test = load_and_split()
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 50)
    print("Avaliação do modelo em produção:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia : {acc:.2%}")
    print(f"F1 (weighted): {f1:.3f}")
    print("=" * 50)

    # Salva localmente — o app Streamlit usa este arquivo
    # Isso evita que o app precise acessar o DagsHub em tempo de execução
    joblib.dump(pipeline, "model.pkl")
    print("\nModelo salvo em model.pkl — pronto para o Streamlit!")


if __name__ == "__main__":
    main()
