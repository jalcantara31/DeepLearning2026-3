"""Implementação modular do modelo LSTM (Long Short-Term Memory) em PyTorch puro."""

from __future__ import annotations

import torch
import torch.nn as nn

from ai.models.base import BaseTimeSeriesForecaster


class LSTMForecaster(BaseTimeSeriesForecaster):
    """
    Rede LSTM multicamada para predição de séries temporais multivariadas.

    Arquitetura:
    1. Camada recorrente `nn.LSTM` (batch_first=True)
    2. `nn.LayerNorm` + `nn.Dropout` sobre a representação temporal final
    3. Cabeça de projeção (`nn.Sequential`) mapeando para `(batch_size, horizon, output_size)`
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
        head_hidden_size: int | None = None,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
            output_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        rnn_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
        )

        feat_dim = self.hidden_size * self.num_directions
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.head_dropout = nn.Dropout(self.dropout)

        proj_dim = head_hidden_size if head_hidden_size is not None else max(32, feat_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(proj_dim, self.horizon * self.output_size),
        )

        self._init_recurrent_weights(self.lstm, is_lstm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, lookback, input_size)
        out, _ = self.lstm(x)
        # Extrai o estado do último passo temporal: (batch_size, feat_dim)
        last_step = out[:, -1, :]
        normed = self.head_dropout(self.layer_norm(last_step))
        pred = self.head(normed)
        return pred.view(-1, self.horizon, self.output_size)

