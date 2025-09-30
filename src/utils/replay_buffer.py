import random
from collections import deque, namedtuple
import numpy as np
import torch


class ReplayBuffer:
    """Buffer de tamanho fixo para armazenar tuplas de experiência."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """
        Inicializa um objeto ReplayBuffer.
        Args:
            buffer_size (int): Tamanho máximo do buffer
            batch_size (int): Tamanho de cada batch de treinamento
            seed (int): Semente para reprodutibilidade
            device (torch.device): Dispositivo (CPU ou GPU)
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Adiciona uma nova experiência à memória."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Amostra aleatoriamente um batch de experiências da memória."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Retorna o tamanho atual da memória."""
        return len(self.memory)
