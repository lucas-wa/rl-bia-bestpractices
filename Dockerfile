# 1. Usar uma imagem oficial do Python como base
# A tag 'slim' é uma versão mais leve, boa para projetos
FROM python:3.9-slim

# 2. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 3. Instalar dependências do sistema necessárias para o gym e renderização
# 'xvfb' e 'ffmpeg' permitem a renderização em um display virtual
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libx11-dev \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

# 4. Copiar o arquivo de dependências e instalar os pacotes Python
# Isso é feito primeiro para aproveitar o cache do Docker. As dependências
# não serão reinstaladas a cada mudança no código-fonte.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o restante do código-fonte do projeto para o container
COPY . .

# 6. Comando padrão (pode ser sobrescrito ao rodar o container)
# Informa ao usuário como usar a imagem.
CMD ["echo", "Imagem pronta. Execute um script, por exemplo: 'python scripts/train.py --agent dqn'"]