"""Fábrica (Factory) e registro modular de modelos PyTorch para séries temporais."""

from __future__ import annotations

from typing import Any, Dict, Type

from ai.models.base import BaseTimeSeriesForecaster
from ai.models.gru import GRUForecaster
from ai.models.lstm import LSTMForecaster

MODEL_REGISTRY: Dict[str, Type[BaseTimeSeriesForecaster]] = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
}


def build_model(
    model_name: str,
    input_size: int,
    horizon: int,
    output_size: int = 1,
    model_cfg: Dict[str, Any] | None = None,
) -> BaseTimeSeriesForecaster:
    """
    Instancia um modelo registrado ('lstm' ou 'gru') a partir do dicionário de configuração.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' não suportado. Modelos disponíveis: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = dict(model_cfg or {})
    # Permite hiperparâmetros específicos por arquitetura no YAML (ex: cfg['lstm']) ou globais
    arch_specific = cfg.get(key, {})
    if isinstance(arch_specific, dict):
        merged = {k: v for k, v in cfg.items() if k not in MODEL_REGISTRY and k != "name" and k != "names"}
        merged.update(arch_specific)
    else:
        merged = {k: v for k, v in cfg.items() if k != "name" and k != "names"}

    model_cls = MODEL_REGISTRY[key]
    return model_cls(
        input_size=input_size,
        hidden_size=int(merged.get("hidden_size", 64)),
        num_layers=int(merged.get("num_layers", 2)),
        horizon=horizon,
        output_size=output_size,
        dropout=float(merged.get("dropout", 0.2)),
        bidirectional=bool(merged.get("bidirectional", False)),
        head_hidden_size=merged.get("head_hidden_size", None),
    )

