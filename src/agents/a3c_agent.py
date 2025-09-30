import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from src.models.a3c_model import ActorCritic


class A3CAgent:
    """Classe que define um worker A3C simplificado."""

    def __init__(self, state_size, action_size, global_model, optimizer, config):
        """
        Inicializa o worker A3C.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device("cpu")

        # Usar apenas modelo local, sem compartilhamento
        self.local_model = ActorCritic(
            state_size, action_size, config["hidden_layers"]
        ).to(self.device)
        self.local_optimizer = optim.Adam(
            self.local_model.parameters(), lr=config["learning_rate"]
        )

    def act(self, state):
        """Retorna uma ação."""
        state = torch.from_numpy(state).float().to(self.device)
        self.local_model.eval()
        with torch.no_grad():
            probs, value = self.local_model(state)
        self.local_model.train()

        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()

        return action.item(), log_prob, entropy, value

    def learn(
        self, states, actions, rewards, next_states, dones, log_probs, entropies, values
    ):
        """Aprende usando apenas modelo local."""
        try:
            # Calcula o retorno (G)
            R = 0
            if not dones[-1]:
                _, R = self.local_model(
                    torch.from_numpy(next_states[-1]).float().to(self.device)
                )
                R = R.detach().cpu().numpy()[0]

            returns = []
            for r in reversed(rewards):
                R = r + self.config["gamma"] * R
                returns.insert(0, R)

            returns = torch.tensor(
                returns, dtype=torch.float32, requires_grad=False
            ).to(self.device)

            # Recalcula com gradientes
            states_tensor = torch.stack(
                [torch.from_numpy(s).float() for s in states]
            ).to(self.device)
            actions_tensor = torch.tensor(actions).to(self.device)

            probs, values_new = self.local_model(states_tensor)
            m = Categorical(probs)
            log_probs_new = m.log_prob(actions_tensor)
            entropies_new = m.entropy()

            # Calcula as perdas
            advantage = returns - values_new.squeeze()
            actor_loss = -(log_probs_new * advantage.detach()).mean()
            critic_loss = F.smooth_l1_loss(returns, values_new.squeeze())
            entropy_loss = -entropies_new.mean()

            loss = (
                actor_loss
                + self.config["value_loss_coef"] * critic_loss
                + self.config["entropy_coef"] * entropy_loss
            )

            # Atualizar modelo local
            self.local_optimizer.zero_grad()
            loss.backward()
            self.local_optimizer.step()

            return actor_loss, critic_loss, entropy_loss, advantage, loss

        except Exception as e:
            print(f"Erro no learn(): {e}")
            raise
