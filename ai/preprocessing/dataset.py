"""Dataset PyTorch de janelas deslizantes (Sliding Window) isolado por partição temporal."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ai.preprocessing.preprocessor import PreparedSplits

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """
    Dataset PyTorch eficiente em memória para predição de séries temporais multivariadas.

    Cada amostra (X, y) é extraída exclusivamente de uma única partição (Treino, Validação ou Teste):
    - X: tensor float32 de formato (lookback, n_features) cobrindo [t - lookback, ..., t - 1]
    - y: tensor float32 de formato (horizon, n_targets) cobrindo [t, ..., t + horizon - 1]

    Qualquer janela que contenha valores ausentes (paradas operacionais ou gaps de historiador)
    é descartada automaticamente de forma vetorizada.
    """

    def __init__(
        self,
        df_scaled: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        lookback: int = 48,
        horizon: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if lookback < 1 or horizon < 1 or stride < 1:
            raise ValueError("lookback, horizon e stride devem ser inteiros >= 1.")

        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.stride = int(stride)

        self.features_arr = np.ascontiguousarray(df_scaled[self.feature_cols].values, dtype=np.float32)
        self.targets_arr = np.ascontiguousarray(df_scaled[self.target_cols].values, dtype=np.float32)
        self.timestamps = df_scaled.index

        total_window = self.lookback + self.horizon
        n_rows = len(df_scaled)
        if n_rows < total_window:
            raise ValueError(
                f"Partição possui apenas {n_rows} linhas, insuficiente para "
                f"lookback ({self.lookback}) + horizon ({self.horizon}) = {total_window}."
            )

        # Identificação vetorizada rápida de janelas 100% livres de NaN
        valid_row = (~np.isnan(self.features_arr).any(axis=1)) & (
            ~np.isnan(self.targets_arr).any(axis=1)
        )
        # Soma cumulativa permite checar em O(1) se todos os pontos em [start, start + total_window) são válidos
        csum = np.concatenate([[0], np.cumsum(valid_row.astype(np.int64))])
        candidate_starts = np.arange(0, n_rows - total_window + 1, self.stride)
        window_valid_counts = csum[candidate_starts + total_window] - csum[candidate_starts]
        self.valid_starts = candidate_starts[window_valid_counts == total_window]

        if len(self.valid_starts) == 0:
            raise ValueError(
                "Nenhuma janela contínua válida encontrada nesta partição. "
                "Reduza lookback/horizon ou ajuste o filtro de pré-processamento."
            )

        # Timestamps correspondentes ao primeiro passo previsto (t = start + lookback)
        self.target_timestamps = self.timestamps[self.valid_starts + self.lookback]

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.valid_starts[idx])
        mid = start + self.lookback
        end = mid + self.horizon

        x = torch.from_numpy(self.features_arr[start:mid])
        y = torch.from_numpy(self.targets_arr[mid:end])
        return x, y


def create_dataloaders(
    splits: PreparedSplits,
    lookback: int = 48,
    horizon: int = 1,
    stride_train: int = 1,
    stride_eval: int = 1,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, Optional[DataLoader]]:
    """
    Cria os DataLoaders de Treino, Validação e (opcionalmente) Teste de forma isolada,
    validando que não existe interseção temporal entre janelas de partições distintas.
    """
    train_ds = SlidingWindowDataset(
        df_scaled=splits.train_scaled,
        feature_cols=splits.feature_cols,
        target_cols=splits.target_cols,
        lookback=lookback,
        horizon=horizon,
        stride=stride_train,
    )
    val_ds = SlidingWindowDataset(
        df_scaled=splits.val_scaled,
        feature_cols=splits.feature_cols,
        target_cols=splits.target_cols,
        lookback=lookback,
        horizon=horizon,
        stride=stride_eval,
    )

    # Garantir que a última janela de treino termina estritamente antes da primeira janela de validação
    train_last_target_ts = train_ds.timestamps[int(train_ds.valid_starts[-1]) + lookback + horizon - 1]
    val_first_input_ts = val_ds.timestamps[int(val_ds.valid_starts[0])]
    assert train_last_target_ts < val_first_input_ts, (
        f"VIOLAÇÃO DE JANELA: Último alvo de treino ({train_last_target_ts}) >= "
        f"primeira entrada de validação ({val_first_input_ts})!"
    )

    test_ds: Optional[SlidingWindowDataset] = None
    if splits.test_scaled is not None and len(splits.test_scaled) >= (lookback + horizon):
        test_ds = SlidingWindowDataset(
            df_scaled=splits.test_scaled,
            feature_cols=splits.feature_cols,
            target_cols=splits.target_cols,
            lookback=lookback,
            horizon=horizon,
            stride=stride_eval,
        )
        val_last_target_ts = val_ds.timestamps[int(val_ds.valid_starts[-1]) + lookback + horizon - 1]
        test_first_input_ts = test_ds.timestamps[int(test_ds.valid_starts[0])]
        assert val_last_target_ts < test_first_input_ts, (
            f"VIOLAÇÃO DE JANELA: Último alvo de validação ({val_last_target_ts}) >= "
            f"primeira entrada de teste ({test_first_input_ts})!"
        )

    use_pin = bool(pin_memory and torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=False,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
            drop_last=False,
        )
        if test_ds is not None
        else None
    )

    logger.info(
        "Janelas construídas (lookback=%d, horizon=%d) | Treino: %d janelas | Validação: %d janelas%s",
        lookback,
        horizon,
        len(train_ds),
        len(val_ds),
        f" | Teste: {len(test_ds)} janelas" if test_ds is not None else "",
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

