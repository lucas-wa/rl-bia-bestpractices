import yaml
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os
import multiprocessing as mp
import sys
import time

# Remover esta linha - usar método padrão (fork)
# mp.set_start_method('spawn', force=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dqn_agent import DQNAgent
from src.agents.a3c_agent import A3CAgent
from src.models.a3c_model import ActorCritic
from src.utils.logger import TensorBoardLogger


def train_dqn(config):
    """Função para treinar o agente DQN."""
    logger = TensorBoardLogger(config["log_dir"])
    env = gym.make(config["env_id"])
    # Remover env.seed() - não existe no Gymnasium
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config)

    scores = []
    scores_window = deque(maxlen=100)
    epsilon = config["epsilon_start"]

    for i_episode in range(1, config["num_episodes"] + 1):
        # Mudança na API do reset() - agora retorna (state, info)
        state, _ = env.reset(seed=config["seed"])
        score = 0
        count_t = 0
        for t in range(config["max_steps_per_episode"]):
            action = agent.act(state, epsilon)
            # Mudança na API do step() - agora retorna (state, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Combinar os dois
            loss = agent.step(state, action, reward, next_state, done)
            logger.log_scalar("DQN Loss", loss, i_episode + t)
            state = next_state
            score += reward
            count_t += 1
            if done:
                break
        logger.log_scalar("Episode Count", count_t, i_episode)
        scores_window.append(score)
        scores.append(score)

        epsilon = max(config["epsilon_end"], config["epsilon_decay"] * epsilon)

        logger.log_scalar("Epsilon", epsilon, i_episode)

        logger.log_scalar("Reward", score, i_episode)
        logger.log_scalar(
            "Average Reward (100 episodes)", np.mean(scores_window), i_episode
        )

        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if i_episode % config["target_update_frequency"] == 0:
            agent.update_target_network()

        if np.mean(scores_window) >= 200.0:
            print(
                f"\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            os.makedirs(config["model_save_path"], exist_ok=True)
            torch.save(
                agent.qnetwork_local.state_dict(),
                os.path.join(config["model_save_path"], "checkpoint.pth"),
            )
            break

    logger.close()
    env.close()


def worker(worker_id, config, episode_counter):
    """Função do worker para A3C."""
    try:
        # CADA WORKER CRIA SEU PRÓPRIO LOGGER
        logger = TensorBoardLogger(config["log_dir"])

        env = gym.make(config["env_id"])
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # CADA WORKER TEM SEU PRÓPRIO AGENTE (sem modelo global compartilhado)
        agent = A3CAgent(state_size, action_size, None, None, config)

        scores_window = deque(maxlen=100)
        episode_count = 0

        while episode_counter.value < config["max_episodes"]:
            try:
                step_counter = 0
                state, _ = env.reset(seed=config["seed"] + worker_id)
                episode_reward = 0

                states, actions, rewards, dones = [], [], [], []
                log_probs, entropies, values = [], [], []

                done = False
                while not done: 
                    action, log_prob, entropy, value = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)
                    log_probs.append(log_prob)
                    entropies.append(entropy)
                    values.append(value)

                    state = next_state
                    step_counter += 1

                actor_loss, critic_loss, entropy_loss, advantage, loss = agent.learn(
                    states,
                    actions,
                    rewards,
                    [next_state],
                    dones,
                    log_probs,
                    entropies,
                    values,
                )


                logger.log_scalar(
                    f"Worker_{worker_id}/train/policy_loss", actor_loss, current_episode
                )
                logger.log_scalar(
                    f"Worker_{worker_id}/train/value_loss", critic_loss, current_episode
                )
                logger.log_scalar(
                    f"Worker_{worker_id}/train/advantage", advantage, current_episode
                )
                # @TODO verificar se é mesmo a entropy_loss
                # logger.log_scalar(
                #     f"Worker_{worker_id}/train/explained_variance",
                #     entropy_loss,
                #     current_episode,
                # )

                logger.log_scalar(
                    f"Worker_{worker_id}/train/entropy_loss",
                    entropy_loss,
                    current_episode,
                )

                logger.log_scalar(
                    f"Worker_{worker_id}/train/step_counter",
                    entropy_loss,
                    current_episode,
                )

                scores_window.append(episode_reward)

                with episode_counter.get_lock():
                    episode_counter.value += 1
                    episode_count += 1
                    current_episode = episode_counter.value

                    # Registra as métricas no TensorBoard
                    logger.log_scalar(
                        f"Worker_{worker_id}/Reward", episode_reward, current_episode
                    )
                    logger.log_scalar("Global/Reward", episode_reward, current_episode)
                    logger.log_scalar(
                        "Global/Average_Reward_100_Episodes",
                        np.mean(scores_window),
                        current_episode,
                    )

                    # Apenas worker 0 imprime para não poluir o console
                    if worker_id == 0 and episode_count % 10 == 0:
                        print(
                            f"Episodes: {current_episode}, Avg Score: {np.mean(scores_window):.2f}"
                        )

            except Exception as e:
                print(f"Erro no episódio do worker {worker_id}: {e}")
                break

        # FECHA O LOGGER E O AMBIENTE DO WORKER
        logger.close()
        env.close()
        print(f"Worker {worker_id} finalizado após {episode_count} episódios")

    except Exception as e:
        print(f"Erro fatal no worker {worker_id}: {e}")
        raise


def train_a3c(config):
    """Função para treinar o agente A3C."""
    print("Iniciando treinamento A3C...")

    env = gym.make(config["env_id"])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    torch.manual_seed(config["seed"])

    device = torch.device("cpu")
    print(f"Usando device: {device}")

    episode_counter = mp.Value("i", 0)

    processes = []
    for i in range(config["num_processes"]):
        # Não passar modelo global nem optimizer
        p = mp.Process(target=worker, args=(i, config, episode_counter))
        p.start()
        processes.append(p)

    print(f"Total de processos: {len(processes)}")
    print("Iniciando treinamento A3C...")

    # Monitorar progresso
    start_time = time.time()
    last_episode = 0

    while any(p.is_alive() for p in processes):
        time.sleep(5)

        current_episode = episode_counter.value
        elapsed_time = time.time() - start_time

        if current_episode > last_episode:
            eps_per_sec = current_episode / elapsed_time if elapsed_time > 0 else 0
            print(
                f"[PROGRESS] {current_episode}/{config['max_episodes']} episódios "
                f"({current_episode/config['max_episodes']*100:.1f}%), "
                f"Eps/sec: {eps_per_sec:.2f}"
            )
            last_episode = current_episode

    for p in processes:
        p.join()
        print(f"Processo finalizado")

    print("\nTraining finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Treina um agente no ambiente LunarLander-v3."
    )
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

    if args.agent == "dqn":
        train_dqn(config)
    elif args.agent == "a3c":
        train_a3c(config)
