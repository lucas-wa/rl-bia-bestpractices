import yaml
import gym
import torch
import numpy as np
import time

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dqn_model import QNetwork
from src.models.a3c_model import ActorCritic


def enjoy(config):
    """
    Visualiza um agente treinado interagindo com o ambiente.
    Args:
        config (dict): Dicionário de configuração.
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

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        time.sleep(0.02)

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if config["agent_name"] == "DQN":
                action_values = model(state_tensor)
                action = np.argmax(action_values.cpu().data.numpy())
            else:  # A3C
                probs, _ = model(state_tensor)
                action = torch.argmax(probs).item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Episódio finalizado. Recompensa total: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualiza um agente treinado.")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "a3c"],
        required=True,
        help="Escolha o agente (dqn ou a3c)",
    )
    args = parser.parse_args()

    config_path = f"configs/{args.agent}_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    enjoy(config)
