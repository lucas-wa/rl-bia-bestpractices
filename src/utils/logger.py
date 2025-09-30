from torch.utils.tensorboard import SummaryWriter
import os
import threading
import queue
import time
import json
from datetime import datetime


class TensorBoardLogger:
    """Classe para registrar métricas no TensorBoard."""

    def __init__(self, log_dir="logs/"):
        """
        Inicializa o logger.
        Args:
            log_dir (str): Diretório para salvar os logs do TensorBoard.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """
        Registra um valor escalar.
        Args:
            tag (str): Nome da métrica (ex: 'Reward', 'Loss').
            value (float): Valor da métrica.
            step (int): Passo de tempo (ex: número do episódio).
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Fecha o writer do TensorBoard."""
        self.writer.close()
