# 💬 Análise de Sentimento — MLOps Completo

App de análise de sentimento de reviews com rastreamento completo de experimentos via **DagsHub + MLflow**, containerizado com **Docker** e publicado no **Render**.

Projeto didático da aula de **Containers, Docker, ML, MLflow e Streamlit**.

---

## Fluxo completo

```
Dados (CSV) → Experimentos (MLflow) → DagsHub (comparar) → Promover modelo → Streamlit → Docker → Render
```

---

## Configuração inicial

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/SEU_USUARIO/sentimento-app.git
cd sentimento-app
pip install -r requirements.txt
```

### 2. Criar o arquivo .env com suas credenciais do DagsHub

```bash
cp .env.example .env
# Edite o .env com seu usuário, repositório e token do DagsHub
```

> Obtenha o token em: dagshub.com → User Settings → Tokens → New Token

---

## Executando os experimentos

### Experimento 1 — baseline

No `src/train.py`, configure:
```python
MAX_FEATURES = 3000
NGRAM_MAX    = 1
C            = 0.1
RUN_NAME     = "exp-1-baseline"
```

```bash
python src/train.py
```

### Experimento 2 — bigramas

```python
MAX_FEATURES = 5000
NGRAM_MAX    = 2
C            = 1.0
RUN_NAME     = "exp-2-bigramas"
```

```bash
python src/train.py
```

### Experimento 3 — vocabulário largo

```python
MAX_FEATURES = 10000
NGRAM_MAX    = 2
C            = 10.0
RUN_NAME     = "exp-3-vocab-largo"
```

```bash
python src/train.py
```

---

## Comparando no DagsHub

1. Acesse `dagshub.com/SEU_USUARIO/sentimento-app`
2. Vá na aba **Experiments**
3. Compare os runs por `f1_weighted`
4. Identifique o melhor modelo

---

## Baixar o melhor modelo para uso local

```bash
python src/evaluate.py
# Gera o arquivo model.pkl
```

---

## Rodar o app localmente

```bash
streamlit run app.py
# Acesse: http://localhost:8501
```

---

## Rodar com Docker (local)

```bash
docker build \
  --build-arg DAGSHUB_USER=seu_usuario \
  --build-arg DAGSHUB_REPO=sentimento-app \
  --build-arg DAGSHUB_TOKEN=seu_token \
  -t sentimento-app .

docker run -p 8501:8501 sentimento-app
```

---

## Deploy no Render

1. Acesse [render.com](https://render.com) → **New → Web Service**
2. Conecte o repositório GitHub
3. Runtime: **Docker** | Port: **8501**
4. Adicione as variáveis de ambiente:
   - `DAGSHUB_USER`
   - `DAGSHUB_REPO`
   - `DAGSHUB_TOKEN`
5. Clique em **Create Web Service**

O Render usa o Dockerfile — o `evaluate.py` baixa automaticamente o modelo mais atual do DagsHub durante o build.

---

## Estrutura do projeto

```
sentimento-app/
├── data/
│   └── reviews.csv          # Dataset de reviews (300 exemplos, 3 classes)
├── src/
│   ├── __init__.py
│   ├── prepare_data.py      # Limpeza e split treino/teste
│   ├── train.py             # Treinamento + MLflow logging
│   └── evaluate.py          # Baixa modelo do DagsHub e salva localmente
├── app.py                   # Interface Streamlit
├── requirements.txt
├── Dockerfile
├── .env.example             # Template de credenciais
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Dataset

300 reviews sintéticos em português com 3 classes balanceadas:
- `positivo` — elogios ao produto/serviço
- `negativo` — reclamações e insatisfações
- `neutro` — avaliações neutras ou ambíguas


git clone https://dagshub.com/fbdn-cid25/incidentes-app.git
cd incidentes-app
touch README.md
git add README.md
git commit -m "Initial commit"
git push -u origin main