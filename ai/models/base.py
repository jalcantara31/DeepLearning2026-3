"""Classe base em PyTorch puro para modelos recorrentes de predição de séries temporais."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTimeSeriesForecaster(nn.Module, ABC):
    """
    Interface base para modelos de predição de séries temporais multivariadas em PyTorch puro.

    Entrada esperada:
        x: Tensor de formato (batch_size, lookback, input_size)
    Saída produzida:
        y_hat: Tensor de formato (batch_size, horizon, output_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 1,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.horizon = int(horizon)
        self.output_size = int(output_size)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

    def count_parameters(self) -> int:
        """Retorna o número total de parâmetros treináveis do modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _init_recurrent_weights(rnn_module: nn.Module, is_lstm: bool = False) -> None:
        """
        Inicialização ortogonal nos pesos recorrentes (hh) e Xavier uniforme nos pesos
        de entrada (ih), melhorando a estabilidade de gradiente em séries longas.
        Para LSTM, inicializa o bias do forget gate em +1.0 para facilitar memória inicial.
        """
        for name, param in rnn_module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                if is_lstm:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o passo forward e retorna tensor (batch_size, horizon, output_size)."""
        raise NotImplementedError

