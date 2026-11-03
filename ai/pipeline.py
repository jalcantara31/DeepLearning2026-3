"""
Ponto de acesso direto ao pipeline de séries temporais (reexporta ai.pipeline.pipeline).
Permite tanto `from ai.pipeline import TimeSeriesPipeline` quanto inspeção direta em `ai/pipeline.py`.
"""

from ai.pipeline.pipeline import (
    TimeSeriesPipeline,
    load_yaml_config,
    resolve_device,
    set_global_seed,
)

__all__ = [
    "TimeSeriesPipeline",
    "load_yaml_config",
    "resolve_device",
    "set_global_seed",
]

