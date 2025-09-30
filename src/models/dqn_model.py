import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Rede Neural para o algoritmo DQN (Q-Network)."""

    def __init__(self, state_size, action_size, hidden_layers):
        """
        Inicializa os parâmetros e constrói o modelo.
        Args:
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            hidden_layers (list of int): Número de nós em cada camada oculta
        """
        super(QNetwork, self).__init__()

        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, action_size)

    def forward(self, state):
        """
        Mapeia o estado para os valores de ação (Q-values).
        Args:
            state (torch.Tensor): O estado de entrada.
        Returns:
            torch.Tensor: Os Q-values para cada ação.
        """
        x = self.network(state)
        return self.output_layer(x)
