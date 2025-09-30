import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Rede Neural para o algoritmo A3C (Actor-Critic)."""

    def __init__(self, state_size, action_size, hidden_layers):
        """
        Inicializa os parâmetros e constrói o modelo.
        Args:
            state_size (int): Dimensão de cada estado
            action_size (int): Dimensão de cada ação
            hidden_layers (list of int): Número de nós na camada oculta compartilhada
        """
        super(ActorCritic, self).__init__()

        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Camada de saída para o Actor (política)
        self.policy_head = nn.Linear(input_size, action_size)

        # Camada de saída para o Critic (valor do estado)
        self.value_head = nn.Linear(input_size, 1)

    def forward(self, state):
        """
        Mapeia o estado para a distribuição de probabilidade de ações e o valor do estado.
        Args:
            state (torch.Tensor): O estado de entrada.
        Returns:
            tuple: (distribuição de probabilidade sobre as ações, valor do estado)
        """
        try:
            x = self.shared_layers(state)

            action_logits = self.policy_head(x)
            action_prob = F.softmax(action_logits, dim=-1)

            state_value = self.value_head(x)

            return action_prob, state_value
        except Exception as e:
            print(f"Erro no forward(): {e}")
            raise
