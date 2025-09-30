import yaml
import gymnasium as gym
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn_model import QNetwork
from src.models.a3c_model import ActorCritic


def evaluate(config, num_episodes=100):
    """
    Avalia um agente treinado.
    Args:
        config (dict): Dicionário de configuração.
        num_episodes (int): Número de episódios para avaliação.
    """
    env = gym.make(config["env_id"])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega o modelo treinado
    if config["agent_name"] == "DQN":
        model = QNetwork(state_size, action_size, config["hidden_layers"]).to(device)
    elif config["agent_name"] == "A3C":
        model = ActorCritic(state_size, action_size, config["hidden_layers"]).to(device)
    else:
        raise ValueError("Agente não suportado.")

    model.load_state_dict(torch.load(f"{config['model_save_path']}/checkpoint.pth"))
    model.eval()

    scores = []

    for i in range(num_episodes):
        # Mudança na API do reset() - agora retorna (state, info)
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if config["agent_name"] == "DQN":
                    action_values = model(state_tensor)
                    action = np.argmax(action_values.cpu().data.numpy())
                else:  # A3C
                    probs, _ = model(state_tensor)
                    action = torch.argmax(probs).item()

            # Mudança na API do step() - agora retorna (state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Combinar os dois
            score += reward
            state = next_state
        scores.append(score)
        print(f"Episode {i+1}/{num_episodes}, Score: {score}")

    env.close()
    print(f"\nResultados da Avaliação:")
    print(
        f"Média de Recompensa em {num_episodes} episódios: {np.mean(scores):.2f} +/- {np.std(scores):.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Avalia um agente treinado.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "a3c"],
        required=True,
        help="Escolha o agente (dqn ou a3c)",
    )
    args = parser.parse_args()

    config_path = f"../configs/{args.agent}_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config)
