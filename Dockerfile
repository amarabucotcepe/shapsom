# Use a imagem oficial do Python 3.11 como base
FROM python:3.11

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os arquivos de requisitos para o container
COPY requirements.txt .

# Instala os pacotes necessários
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do aplicativo para o container
COPY . .

# Expõe a porta que o aplicativo usa
EXPOSE 8501

# Executa o aplicativo
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "Página_Inicial.py"]
