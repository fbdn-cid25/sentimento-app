"""
prepare_data.py — Carrega, limpa e divide o dataset de reviews.

Uso standalone:
    python src/prepare_data.py

Uso como módulo (importado pelo train.py e evaluate.py):
    from src.prepare_data import load_and_split
"""
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
   path: str = str(BASE_DIR / "data" / "reviews.csv"),
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Carrega o CSV de reviews, faz limpeza básica e divide em treino/teste.

    Parâmetros:
        path      : caminho para o arquivo CSV
        test_size : proporção do conjunto de teste (padrão: 20%)
        seed      : semente aleatória para reprodutibilidade

    Retorna:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)

    # Remove linhas sem texto ou sem rótulo
    df = df.dropna(subset=["texto", "sentimento"])

    # Remove linhas com texto vazio
    df = df[df["texto"].str.strip() != ""]

    # Normalização básica do texto (minúsculas + remover espaços extras)
    df["texto"] = df["texto"].str.lower().str.strip()

    X = df["texto"]
    y = df["sentimento"]

    # stratify=y garante que a proporção de classes seja igual em treino e teste
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Total de amostras de treino : {len(X_train)}")
    print(f"Total de amostras de teste  : {len(X_test)}")
    print(f"\nDistribuição de classes no treino:")
    print(y_train.value_counts())
