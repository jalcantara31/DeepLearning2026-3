#!/usr/bin/env python3
"""
Entrypoint de linha de comando (CLI) via `argparse` para execução do pipeline
de predição de séries temporais (GRU e LSTM) sobre `data/compressor.csv`.

Exemplos de uso suportados:
    python ai/run.py --config ai/config.yaml
    python ai/run.py --"ai/config.yaml"
    python run.py --config ai/config.yaml
    python run.py --"ai/config.yaml"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Garante que a raiz do repositório esteja no sys.path mesmo rodando de dentro de ai/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.pipeline.pipeline import TimeSeriesPipeline, load_yaml_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline Modular de Predição de Séries Temporais (GRU & LSTM) — Sem Data Leakage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Caminho para o arquivo de configuração .yaml (ex.: ai/config.yaml).",
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        type=str,
        default=None,
        help="Caminho posicional opcional para o arquivo .yaml (alternativa a --config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["gru", "lstm", "both"],
        help="Sobrescreve o(s) modelo(s) definido(s) no YAML ('gru', 'lstm' ou 'both').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescreve o número de épocas de treinamento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sobrescreve o tamanho do lote (batch_size).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Sobrescreve a taxa de aprendizado (learning rate).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Sobrescreve o dispositivo ('auto', 'cuda' ou 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobrescreve a semente global de reprodutibilidade.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Sobrescreve o diretório de saída dos resultados.",
    )
    return parser


def resolve_config_arg(
    args: argparse.Namespace, unknown_args: List[str]
) -> str:
    """
    Resolve o caminho do arquivo YAML suportando:
    1. `--config caminho.yaml` / `-c caminho.yaml`
    2. `python run.py --"caminho.yaml"` (onde o próprio caminho vem prefixado por `--`)
    3. `python run.py caminho.yaml` (argumento posicional)
    """
    if args.config:
        return args.config
    if args.config_pos:
        return args.config_pos

    for token in unknown_args:
        if token.startswith("--") and len(token) > 2:
            candidate = token[2:].strip().strip('"').strip("'")
            if candidate.lower().endswith((".yaml", ".yml")) or Path(candidate).exists():
                return candidate
        elif token.startswith("-") and len(token) > 1:
            candidate = token[1:].strip().strip('"').strip("'")
            if candidate.lower().endswith((".yaml", ".yml")) or Path(candidate).exists():
                return candidate

    default_cfg = REPO_ROOT / "ai" / "config.yaml"
    if default_cfg.is_file():
        return str(default_cfg)

    raise SystemExit(
        "Erro: informe o caminho do arquivo YAML via `python run.py --config ai/config.yaml` "
        "ou `python run.py --\"ai/config.yaml\"`."
    )


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Aplica sobrescritas opcionais de linha de comando sobre o dicionário carregado do YAML."""
    cfg.setdefault("experiment", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})

    if args.model is not None:
        if args.model == "both":
            cfg["model"]["names"] = ["gru", "lstm"]
            cfg["model"].pop("name", None)
        else:
            cfg["model"]["name"] = args.model
            cfg["model"].pop("names", None)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.device is not None:
        cfg["experiment"]["device"] = args.device
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir

    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    config_path = resolve_config_arg(args, unknown)

    cfg = load_yaml_config(config_path)
    cfg = apply_cli_overrides(cfg, args)

    pipeline = TimeSeriesPipeline(cfg)
    report = pipeline.run()

    print("\n" + "=" * 80)
    print("RESUMO DE DESEMPENHO DOS MODELOS (ESCALA FÍSICA ORIGINAL — SEM DATA LEAKAGE)")
    print("=" * 80)
    for model_name, m_info in report["models"].items():
        val_m = m_info["val"]["metrics_physical"]
        print(
            f"[{model_name.upper():4s} | VAL ] Melhor Época: {m_info['best_epoch']:02d} | "
            f"RMSE: {val_m['RMSE']:.4f} | MAE: {val_m['MAE']:.4f} | "
            f"MAPE: {val_m['MAPE_pct']:.2f}% | R²: {val_m['R2']:.4f}"
        )
        if m_info.get("test") is not None:
            test_m = m_info["test"]["metrics_physical"]
            print(
                f"[{model_name.upper():4s} | TEST] Melhor Época: {m_info['best_epoch']:02d} | "
                f"RMSE: {test_m['RMSE']:.4f} | MAE: {test_m['MAE']:.4f} | "
                f"MAPE: {test_m['MAPE_pct']:.2f}% | R²: {test_m['R2']:.4f}"
            )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

