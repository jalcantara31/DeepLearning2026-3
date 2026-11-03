"""
Orquestrador principal do pipeline modular de predição de séries temporais (LSTM e GRU)
sem Data Leakage em PyTorch puro.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from ai.evaluation.evaluator import ModelEvaluator
from ai.loader.data_loader import CompressorDataLoader
from ai.models.factory import build_model
from ai.preprocessing.dataset import create_dataloaders
from ai.preprocessing.preprocessor import TimeSeriesPreprocessor
from ai.training.trainer import PyTorchTrainer

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Fixa sementes globais de Python, NumPy e PyTorch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve o dispositivo PyTorch ('auto', 'cuda' ou 'cpu')."""
    key = device_str.lower().strip()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA solicitado mas não disponível; utilizando CPU.")
        return torch.device("cpu")
    return torch.device(key)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Carrega e valida com segurança (yaml.safe_load) o arquivo de configuração .yaml.
    """
    raw = Path(config_path)
    repo_root = Path(__file__).resolve().parents[2]
    ai_root = Path(__file__).resolve().parents[1]

    candidates = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.extend([
            (Path.cwd() / raw).resolve(),
            (repo_root / raw).resolve(),
            (ai_root / raw).resolve(),
            (ai_root / "configs" / raw.name).resolve(),
        ])

    resolved: Path | None = None
    for cand in candidates:
        if cand.is_file() and cand.suffix.lower() in (".yaml", ".yml"):
            resolved = cand
            break

    if resolved is None:
        raise FileNotFoundError(
            f"Arquivo de configuração YAML não encontrado: '{config_path}'. "
            f"Caminhos verificados: {[str(c) for c in candidates]}"
        )

    with open(resolved, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"O arquivo YAML '{resolved}' deve conter um dicionário na raiz.")

    cfg["_config_path"] = str(resolved)
    return cfg


class TimeSeriesPipeline:
    """
    Pipeline completo e modular de predição de séries temporais para os sensores do compressor.

    Etapas executadas:
    1. Leitura e padronização do CSV (`CompressorDataLoader`);
    2. Regularização temporal, filtro físico instantâneo, divisão cronológica estrita
       (Treino / Validação / Teste) e escalonamento exclusivo no treino (`TimeSeriesPreprocessor`);
    3. Construção de janelas deslizantes isoladas por partição (`create_dataloaders`);
    4. Instanciação e treinamento dos modelos PyTorch (`GRU` e/ou `LSTM`) via `PyTorchTrainer`;
    5. Avaliação na escala física original, geração de gráficos e exportação de relatórios (`ModelEvaluator`).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[2]

        exp_cfg = self.config.get("experiment", {})
        self.seed = int(exp_cfg.get("seed", 42))
        self.device = resolve_device(str(exp_cfg.get("device", "auto")))

        out_dir_raw = exp_cfg.get("output_dir", "results/ai")
        out_path = Path(out_dir_raw)
        self.output_dir = (
            out_path.resolve() if out_path.is_absolute() else (self.repo_root / out_path).resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_model_names(self) -> List[str]:
        model_cfg = self.config.get("model", {})
        if isinstance(model_cfg, str):
            return [model_cfg.lower().strip()]

        names = model_cfg.get("names") or model_cfg.get("name") or ["gru", "lstm"]
        if isinstance(names, str):
            if names.lower().strip() in ("all", "both"):
                return ["gru", "lstm"]
            return [names.lower().strip()]
        return [str(n).lower().strip() for n in names]

    def run(self) -> Dict[str, Any]:
        """Executa todas as etapas do pipeline e retorna o relatório consolidado."""
        set_global_seed(self.seed)

        exp_cfg = self.config.get("experiment", {})
        data_cfg = self.config.get("data", {})
        prep_cfg = self.config.get("preprocessing", {})
        win_cfg = self.config.get("windowing", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        # 1. Carregamento dos dados
        loader = CompressorDataLoader(
            csv_path=data_cfg.get("csv_path", "data/compressor.csv"),
            reconstruct_poly_eff_dev=bool(data_cfg.get("reconstruct_poly_eff_dev", True)),
            drop_all_nan_cols=bool(data_cfg.get("drop_all_nan_cols", True)),
        )
        df_raw = loader.load(nrows=data_cfg.get("nrows", None))

        # 2. Pré-processamento e divisão cronológica sem Data Leakage
        preprocessor = TimeSeriesPreprocessor(
            feature_cols=prep_cfg.get("feature_cols", None),
            target_cols=prep_cfg.get("target_cols", "act_poly_eff"),
            resample_freq=prep_cfg.get("resample_freq", "5min"),
            filter_physical_anomalies=bool(prep_cfg.get("filter_physical_anomalies", True)),
            min_power_kw=float(prep_cfg.get("min_power_kw", 4000.0)),
            min_press_ratio=float(prep_cfg.get("min_press_ratio", 1.05)),
            min_eff_pct=float(prep_cfg.get("min_eff_pct", 50.0)),
            max_eff_pct=float(prep_cfg.get("max_eff_pct", 100.0)),
            train_ratio=float(prep_cfg.get("train_ratio", 0.70)),
            val_ratio=float(prep_cfg.get("val_ratio", 0.15)),
            test_ratio=float(prep_cfg.get("test_ratio", 0.15)),
            split_gap_steps=int(prep_cfg.get("split_gap_steps", 1)),
            max_ffill_steps=int(prep_cfg.get("max_ffill_steps", 3)),
            scaler_type=str(prep_cfg.get("scaler_type", "standard")),
        )
        splits = preprocessor.process(df_raw)

        # 3. Criação dos DataLoaders de janelas deslizantes isolados por split
        lookback = int(win_cfg.get("lookback", 48))
        horizon = int(win_cfg.get("horizon", 1))
        batch_size = int(train_cfg.get("batch_size", 128))

        dataloaders = create_dataloaders(
            splits=splits,
            lookback=lookback,
            horizon=horizon,
            stride_train=int(win_cfg.get("stride_train", 1)),
            stride_eval=int(win_cfg.get("stride_eval", 1)),
            batch_size=batch_size,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
        )

        # 4. Treinamento e avaliação dos modelos selecionados (GRU e/ou LSTM)
        model_names = self._resolve_model_names()
        input_size = len(splits.feature_cols)
        output_size = len(splits.target_cols)

        evaluator = ModelEvaluator(
            target_scaler=splits.target_scaler,
            target_cols=splits.target_cols,
            base_output_dir=self.output_dir,
            device=self.device,
            plot_format=exp_cfg.get("plot_format", "pdf"),
        )

        models_results: Dict[str, Any] = {}
        comparison_rows: List[Dict[str, Any]] = []

        for model_name in model_names:
            set_global_seed(self.seed)
            model_dir = evaluator.get_model_output_dir(model_name)
            model_ckpt_dir = model_dir / "checkpoints"
            model_ckpt_dir.mkdir(parents=True, exist_ok=True)

            model = build_model(
                model_name=model_name,
                input_size=input_size,
                horizon=horizon,
                output_size=output_size,
                model_cfg=model_cfg if isinstance(model_cfg, dict) else {},
            )

            trainer = PyTorchTrainer(
                model=model,
                model_name=model_name,
                device=self.device,
                epochs=int(train_cfg.get("epochs", 25)),
                lr=float(train_cfg.get("lr", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                loss_fn_name=str(train_cfg.get("loss", "huber")),
                optimizer_name=str(train_cfg.get("optimizer", "adamw")),
                scheduler_name=str(train_cfg.get("scheduler", "reduce_on_plateau")),
                grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 6)),
                early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 1e-5)),
                checkpoint_dir=model_ckpt_dir,
            )

            history = trainer.fit(
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
            )
            evaluator.plot_training_history(history)

            val_eval = evaluator.evaluate_split(
                model=model,
                model_name=model_name,
                split_name="val",
                loader=dataloaders["val"],
            )
            comparison_rows.append({
                "model": model_name,
                "split": "val",
                "best_epoch": history.best_epoch,
                "best_val_loss_scaled": round(history.best_val_loss, 6),
                **val_eval["metrics_physical"],
            })

            test_eval = None
            if dataloaders["test"] is not None:
                test_eval = evaluator.evaluate_split(
                    model=model,
                    model_name=model_name,
                    split_name="test",
                    loader=dataloaders["test"],
                )
                comparison_rows.append({
                    "model": model_name,
                    "split": "test",
                    "best_epoch": history.best_epoch,
                    "best_val_loss_scaled": round(history.best_val_loss, 6),
                    **test_eval["metrics_physical"],
                })

            models_results[model_name] = {
                "num_parameters": model.count_parameters(),
                "best_epoch": history.best_epoch,
                "best_val_loss_scaled": history.best_val_loss,
                "checkpoint_path": history.checkpoint_path,
                "val": val_eval,
                "test": test_eval,
            }

        # 5. Consolidação e exportação do comparativo final
        comparison_df = pd.DataFrame(comparison_rows)
        comp_csv_path = self.output_dir / "metrics_comparison.csv"
        comparison_df.to_csv(comp_csv_path, index=False)
        evaluator.plot_models_comparison(comparison_df)

        report = {
            "config_path": self.config.get("_config_path"),
            "device": str(self.device),
            "seed": self.seed,
            "feature_cols": splits.feature_cols,
            "target_cols": splits.target_cols,
            "lookback": lookback,
            "horizon": horizon,
            "leakage_audit": splits.leakage_audit,
            "models": models_results,
        }

        report_path = self.output_dir / "pipeline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Pipeline finalizado com sucesso! Resultados salvos em: %s", self.output_dir)
        return report

