"""Módulo de pré-processamento sem data leakage e geração de janelas deslizantes."""

from ai.preprocessing.preprocessor import PreparedSplits, TimeSeriesPreprocessor

try:
    from ai.preprocessing.dataset import SlidingWindowDataset, create_dataloaders
except ImportError:  # pragma: no cover - caso torch ainda não esteja instalado no venv
    SlidingWindowDataset = None  # type: ignore[assignment,misc]
    create_dataloaders = None  # type: ignore[assignment]

__all__ = [
    "PreparedSplits",
    "SlidingWindowDataset",
    "TimeSeriesPreprocessor",
    "create_dataloaders",
]
