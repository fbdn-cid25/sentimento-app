"""
train.py — Treina o modelo de análise de sentimento e registra no MLflow/DagsHub.

Execute UMA VEZ por experimento, mudando os parâmetros marcados com # MUDE AQUI:
    python src/train.py

Experimento 1 (baseline)  : MAX_FEATURES=3000, NGRAM_MAX=1, C=0.1
Experimento 2 (bigramas)  : MAX_FEATURES=5000, NGRAM_MAX=2, C=1.0
Experimento 3 (vocab largo): MAX_FEATURES=10000, NGRAM_MAX=2, C=10.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_and_split  # sem o prefixo src.

import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dotenv import load_dotenv

# Importa a função de preparação de dados do nosso módulo
from src.prepare_data import load_and_split

# ── Carrega as credenciais do arquivo .env ───────────────────────────────────
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# ── Aponta o MLflow para o servidor do DagsHub ───────────────────────────────
# O DagsHub oferece um servidor MLflow gratuito para cada repositório.
# A URL segue o padrão: https://dagshub.com/USUARIO/REPOSITORIO.mlflow
mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)

# Autenticação via variáveis de ambiente (não coloque credenciais no código!)
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Nome do experimento no MLflow (agrupa todos os runs)
mlflow.set_experiment("analise-sentimento")


# ════════════════════════════════════════════════════════════════════════════
# MUDE ESTES PARÂMETROS A CADA EXPERIMENTO
# ════════════════════════════════════════════════════════════════════════════
MAX_FEATURES = 10000     # Quantas palavras/termos o TF-IDF vai considerar
NGRAM_MAX    = 2        # 1 = unigramas | 2 = uni + bigramas
C            = 10.0      # Parâmetro de regularização da Regressão Logística
                        # C pequeno = mais regularização (modelo simples)
                        # C grande  = menos regularização (modelo complexo)
RUN_NAME     = "exp-3-vocab-largo"   # MUDE a cada run para identificar no DagsHub!
# ════════════════════════════════════════════════════════════════════════════


def main():
    # Carrega e divide os dados
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    # Inicia um run no MLflow
    # Tudo dentro do bloco 'with' é rastreado automaticamente
    with mlflow.start_run(run_name=RUN_NAME):

        # 1. Loga os parâmetros do experimento
        # Isso permite comparar configurações diferentes no DagsHub
        mlflow.log_params({
            "max_features": MAX_FEATURES,
            "ngram_range":  f"(1,{NGRAM_MAX})",
            "C":            C,
            "solver":       "lbfgs",
            "test_size":    0.2,
            "dataset":      "data/reviews.csv",
        })

        # 2. Cria o pipeline: TF-IDF → Regressão Logística
        # O Pipeline garante que o mesmo pré-processamento é aplicado
        # tanto no treino quanto na inferência
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=(1, NGRAM_MAX),
                strip_accents="unicode",  # normaliza acentos
                sublinear_tf=True,        # aplica log ao TF para suavizar
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",  # suporte a 3 classes
            )),
        ])

        # 3. Treina o modelo
        pipeline.fit(X_train, y_train)

        # 4. Avalia no conjunto de teste
        y_pred = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")

        print("\n" + "=" * 50)
        print(classification_report(y_test, y_pred))
        print("=" * 50)

        # 5. Loga as métricas no MLflow
        mlflow.log_metrics({
            "accuracy":    acc,
            "f1_weighted": f1,
        })

        # 6. Registra o modelo no MLflow Model Registry
        # registered_model_name cria/atualiza o modelo no registro central
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="SentimentClassifier",
        )

        print(f"\nRun '{RUN_NAME}' finalizado com sucesso!")
        print(f"Acurácia : {acc:.2%}")
        print(f"F1 (weighted): {f1:.3f}")
        print(f"\nVisualize em: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")


if __name__ == "__main__":
    main()
