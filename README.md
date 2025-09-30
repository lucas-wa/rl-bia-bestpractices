# RL Bestpractices Repository

[![BIA](https://img.shields.io/badge/BIA-Inteligência%20Artificial-orange?style=for-the-badge&logo=robot&logoColor=white)](https://bia.ceia.ufes.br/) [![CEIA](https://img.shields.io/badge/CEIA-Centro%20de%20Excelência%20em%20IA-blue?style=for-the-badge&logo=university&logoColor=white)](https://ceia.ufes.br/)
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-DQN%20%7C%20A3C-blue?style=for-the-badge&logo=python&logoColor=white)](https://github.com/pedroamsaraiva/rl-BIA-bestpractices)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

## Instalação

Primeiro, vamos começar baixando o repositorio.

```bash
git clone  https://github.com/PedroMSaraiva/rl-bia-bestpractices.git
```

Você pode conseguir as imagens dos container, buildando a imagem do zero ou puxando do DockerHub(Recomendado)

### DockerHub (Recomendado)

```bash
docker pull pedroamsaraiva/rl-bia-bestprac
```

### Buildando do Zero 
```bash
# execute dentro do repositorio
docker build -t rl-bia-bestpractices .
```

## Entrando no container

Vamos entrar no container para iniciar os trabalhos!

```
docker run -it --rm -p 6006:6006 --gpus all -v "$(pwd):/app" <nome_da_imagem> /bin/bash
```

Otimo, agora pelo terminal você pode executar os comandos

```
cd /scripts/
python3 train --agent dqn

# e depois
python3 train --agent a3c
```

Abra um outro terminal com:
```
docker exec -it <nome_do_container> /bin/bash
```
Se você não souber o nome do container aberte `tab` que ele completa, ou execute `docker ps`

Nesse novo terminal, vamos executar o Tensorboard para visualizar!
```
tensorboard --logdir scripts/results/logs --bind_all
# Se atente ao path que passamos
```

Abra o `localhost:6006` e veja o treinamento.

## Estrutura do repositorio
```text
.
├── configs # Configurações do DQN e A3C
├── Dockerfile
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── enjoy.py
│   ├── evaluate.py
│   ├── results
│   │   ├── logs # Logs salvos
│   │   └── models # Checkpoints
│   └── train.py 
├── src
│   ├── agents
│   │   ├── a3c_agent.py
│   │   ├── dqn_agent.py
│   ├── __init__.py
│   ├── models
│   │   ├── a3c_model.py
│   │   ├── dqn_model.py
│   └── utils
│       ├── logger.py
│       └── replay_buffer.py
└── uv.lock
```
---

Se você fizer um docker-compose, ou otimizar essa imagem/container, me avise(ou crie uma issue) para disponibilizar a todos
Se houver qualquer erro grave, me avise para arrumar.