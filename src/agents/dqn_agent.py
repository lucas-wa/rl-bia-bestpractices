import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from src.models.dqn_model import QNetwork
from src.utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """Agente que interage e aprende com o ambiente usando DQN."""

    def __init__(self, state_size, action_size, config):
        """
        Inicializa um objeto DQNAgent.
        Args:
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            config (dict): Dicionário de configuração
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Redes Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, config["hidden_layers"]
        ).to(self.device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, config["hidden_layers"]
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=config["learning_rate"]
        )

        # Replay memory
        self.memory = ReplayBuffer(
            config["buffer_size"], config["batch_size"], config["seed"], self.device
        )

        # Inicializa o tempo para atualização da rede target
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Salva a experiência no replay buffer e aprende."""
        self.memory.add(state, action, reward, next_state, done)

        # Aprende se houver amostras suficientes na memória
        if len(self.memory) > self.config["batch_size"]:
            experiences = self.memory.sample()
            return self.learn(experiences)

    def act(self, state, eps=0.0):
        """
        Retorna ações para um dado estado de acordo com a política atual.
        Args:
            state (array_like): estado atual
            eps (float): epsilon, para a política epsilon-greedy
        Returns:
            int: ação escolhida
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Seleção Epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """
        Atualiza os pesos da rede neural usando um batch de experiências.
        Args:
            experiences (Tuple[torch.Tensor]): tupla de (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        # Obtém os Q-values esperados da rede target
        q_targets_next = (
            self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # Calcula o Q-target para o estado atual
        q_targets = rewards + (self.config["gamma"] * q_targets_next * (1 - dones))

        # Obtém os Q-values esperados da rede local
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calcula a perda
        loss = F.mse_loss(q_expected, q_targets)

        # Minimiza a perda
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def update_target_network(self):
        """Copia os pesos da rede local para a rede target."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
