FROM python:3.11-slim

WORKDIR /app

# Instala dependências antes de copiar o código (melhor uso do cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto
COPY . .

# Credenciais do DagsHub passadas como build args
# Nunca ficam gravadas na imagem final (ARG é descartado após o build)
ARG DAGSHUB_USER
ARG DAGSHUB_REPO
ARG DAGSHUB_TOKEN

# Disponibiliza as credenciais como variáveis de ambiente durante o build
ENV DAGSHUB_USER=$DAGSHUB_USER
ENV DAGSHUB_REPO=$DAGSHUB_REPO
ENV DAGSHUB_TOKEN=$DAGSHUB_TOKEN

# Baixa o modelo mais recente registrado no DagsHub e salva como model.pkl
# Isso garante que a imagem sempre contém o modelo promovido para produção
RUN python -m src.evaluate

# Expõe a porta do Streamlit
EXPOSE 8501

# 0.0.0.0 é obrigatório para o Render (e qualquer cloud) rotear o tráfego
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
