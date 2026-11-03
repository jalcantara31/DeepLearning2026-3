"""Avaliação de modelos de séries temporais, métricas na escala física original e geração de relatórios."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader

from ai.training.trainer import TrainingHistory

logger = logging.getLogger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas clássicas de predição de séries temporais sobre arrays (N, ...):
    MAE, MSE, RMSE, MAPE (%), sMAPE (%) e R².
    """
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)

    err = yp - yt
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))

    denom_mape = np.where(np.abs(yt) < 1e-6, np.nan, np.abs(yt))
    mape = float(np.nanmean(abs_err / denom_mape) * 100.0)

    denom_smape = (np.abs(yt) + np.abs(yp)) / 2.0
    denom_smape = np.where(denom_smape < 1e-6, np.nan, denom_smape)
    smape = float(np.nanmean(abs_err / denom_smape) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MSE": round(mse, 6),
        "MAPE_pct": round(mape, 4),
        "sMAPE_pct": round(smape, 4),
        "R2": round(r2, 6),
    }


class ModelEvaluator:
    """
    Executa inferência sem vazamento de dados, reverte o escalonamento com o
    target_scaler do treino e salva métricas, predições e gráficos diagnósticos
    em PDF individuais por modelo.
    """

    def __init__(
        self,
        target_scaler: StandardScaler | RobustScaler | MinMaxScaler,
        target_cols: List[str],
        base_output_dir: str | Path,
        device: torch.device,
        plot_format: str = "pdf",
    ) -> None:
        self.target_scaler = target_scaler
        self.target_cols = list(target_cols)
        self.base_output_dir = Path(base_output_dir).resolve()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.plot_format = plot_format.lower().lstrip(".")

    def get_model_output_dir(self, model_name: str) -> Path:
        """
        Garante um diretório exclusivo para cada modelo:
        ex.: results/ai/gru/ ou results/ai/lstm/
        """
        model_dir = self.base_output_dir / model_name.lower().strip()
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _inverse_target(self, arr_3d: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização de um tensor (N, horizon, n_targets) para a escala física original.
        """
        n_samples, horizon, n_targets = arr_3d.shape
        flat = arr_3d.reshape(-1, n_targets)
        inv_flat = self.target_scaler.inverse_transform(flat)
        return inv_flat.reshape(n_samples, horizon, n_targets)

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna (y_true_scaled, y_pred_scaled, y_true_orig, y_pred_orig),
        todos com formato (N, horizon, n_targets).
        """
        model.eval()
        preds_list: List[np.ndarray] = []
        trues_list: List[np.ndarray] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            out = model(x_batch).cpu().numpy()
            preds_list.append(out)
            trues_list.append(y_batch.numpy())

        y_pred_scaled = np.concatenate(preds_list, axis=0)
        y_true_scaled = np.concatenate(trues_list, axis=0)

        y_pred_orig = self._inverse_target(y_pred_scaled)
        y_true_orig = self._inverse_target(y_true_scaled)
        return y_true_scaled, y_pred_scaled, y_true_orig, y_pred_orig

    def evaluate_split(
        self,
        model: nn.Module,
        model_name: str,
        split_name: str,
        loader: DataLoader,
        save_predictions: bool = True,
        save_plots: bool = True,
    ) -> Dict[str, object]:
        """
        Avalia um modelo em uma partição ('val' ou 'test'), calcula métricas na escala
        escalonada e na escala física original, e salva CSV e PDFs diagnósticos individuais.
        """
        model_dir = self.get_model_output_dir(model_name)
        y_true_sc, y_pred_sc, y_true_phys, y_pred_phys = self.predict(model, loader)

        metrics_scaled = compute_regression_metrics(y_true_sc, y_pred_sc)
        metrics_physical = compute_regression_metrics(y_true_phys, y_pred_phys)

        per_target: Dict[str, Dict[str, float]] = {}
        for idx, col in enumerate(self.target_cols):
            per_target[col] = compute_regression_metrics(
                y_true_phys[:, :, idx], y_pred_phys[:, :, idx]
            )

        logger.info(
            "Avaliação [%s | %s] (Escala Física Original) -> RMSE: %.4f | MAE: %.4f | MAPE: %.2f%% | R²: %.4f",
            model_name.upper(),
            split_name.upper(),
            metrics_physical["RMSE"],
            metrics_physical["MAE"],
            metrics_physical["MAPE_pct"],
            metrics_physical["R2"],
        )

        # Extrair timestamps da partição a partir do SlidingWindowDataset
        ds = loader.dataset
        timestamps = getattr(ds, "target_timestamps", pd.RangeIndex(len(y_true_phys)))

        if save_predictions:
            pred_df = pd.DataFrame(index=timestamps)
            for idx, col in enumerate(self.target_cols):
                pred_df[f"{col}_real"] = y_true_phys[:, 0, idx]
                pred_df[f"{col}_pred_{model_name.lower()}"] = y_pred_phys[:, 0, idx]
                pred_df[f"{col}_erro"] = y_pred_phys[:, 0, idx] - y_true_phys[:, 0, idx]

            csv_out = model_dir / f"predictions_{split_name.lower()}.csv"
            pred_df.to_csv(csv_out)

        saved_plots: Dict[str, str] = {}
        if save_plots:
            saved_plots = self.plot_individual_predictions(
                timestamps=timestamps,
                y_true=y_true_phys[:, 0, 0],
                y_pred=y_pred_phys[:, 0, 0],
                target_name=self.target_cols[0],
                model_name=model_name,
                split_name=split_name,
            )

        return {
            "model": model_name,
            "split": split_name,
            "num_windows": int(len(y_true_phys)),
            "metrics_physical": metrics_physical,
            "metrics_scaled": metrics_scaled,
            "per_target_physical": per_target,
            "plots": saved_plots,
        }

    def plot_training_history(self, history: TrainingHistory) -> Dict[str, Path]:
        """
        Salva gráficos individuais em PDF para cada dimensão do treinamento:
        1. loss_train_val.pdf: Curva de Loss Treino vs Validação
        2. learning_rate.pdf: Evolução da Taxa de Aprendizado (LR)
        """
        model_dir = self.get_model_output_dir(history.model_name)
        epochs = np.arange(1, len(history.train_loss) + 1)
        saved = {}

        # 1. Gráfico individual: Loss Treino vs Validação
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(epochs, history.train_loss, label="Treino (Loss)", color="#1f77b4", linewidth=2.0)
        ax.plot(epochs, history.val_loss, label="Validação (Loss)", color="#d62728", linewidth=2.0)
        if history.best_epoch > 0:
            ax.axvline(
                history.best_epoch,
                color="seagreen",
                linestyle="--",
                linewidth=1.5,
                label=f"Melhor Época ({history.best_epoch})",
            )
        ax.set_title(f"Curva de Perda (Loss) — {history.model_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Época", fontsize=10)
        ax.set_ylabel("Loss (Huber / Escala Normalizada)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = model_dir / f"loss_train_val.{self.plot_format}"
        fig.savefig(loss_path, format=self.plot_format)
        plt.close(fig)
        saved["loss"] = loss_path

        # 2. Gráfico individual: Learning Rate
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.plot(epochs, history.lr, color="#9467bd", marker="o", markersize=3.5, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title(f"Evolução da Taxa de Aprendizado (LR) — {history.model_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Época", fontsize=10)
        ax.set_ylabel("Learning Rate (log)", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_path = model_dir / f"learning_rate.{self.plot_format}"
        fig.savefig(lr_path, format=self.plot_format)
        plt.close(fig)
        saved["lr"] = lr_path

        return saved

    def plot_individual_predictions(
        self,
        timestamps: pd.Index,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
        model_name: str,
        split_name: str,
    ) -> Dict[str, str]:
        """
        Gera e salva em PDF individual (sem mesclar):
        1. series_{split}.pdf: Série Temporal Completa Real vs Predito
        2. zoom_{split}.pdf: Zoom Detalhado nas Primeiras Janelas
        3. parity_{split}.pdf: Diagrama de Dispersão de Paridade 1:1
        4. residuals_{split}.pdf: Análise de Erros Residuais ao Longo do Tempo
        """
        model_dir = self.get_model_output_dir(model_name)
        s_name = split_name.lower().strip()
        saved = {}

        # 1. Série Temporal Completa
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        ax.plot(timestamps, y_true, label="Valor Real", color="#1f77b4", linewidth=1.1, alpha=0.85)
        ax.plot(timestamps, y_pred, label=f"Predito ({model_name.upper()})", color="#ff7f0e", linewidth=1.1, alpha=0.85)
        ax.set_title(f"Série Temporal — {model_name.upper()} ({split_name.upper()}) | {target_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestamp", fontsize=10)
        ax.set_ylabel(target_name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p1 = model_dir / f"series_{s_name}.{self.plot_format}"
        fig.savefig(p1, format=self.plot_format)
        plt.close(fig)
        saved["series"] = str(p1)

        # 2. Zoom de Aderência
        n_zoom = min(250, len(y_true))
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        ax.plot(timestamps[:n_zoom], y_true[:n_zoom], label="Valor Real", color="#1f77b4", linewidth=1.8, marker="o", markersize=2.5)
        ax.plot(
            timestamps[:n_zoom],
            y_pred[:n_zoom],
            label=f"Predito ({model_name.upper()})",
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            marker="x",
            markersize=3.0,
        )
        ax.set_title(f"Zoom Detalhado ({n_zoom} passos) — {model_name.upper()} ({split_name.upper()})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestamp", fontsize=10)
        ax.set_ylabel(target_name, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = model_dir / f"zoom_{s_name}.{self.plot_format}"
        fig.savefig(p2, format=self.plot_format)
        plt.close(fig)
        saved["zoom"] = str(p2)

        # 3. Dispersão de Paridade 1:1
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(y_true, y_pred, s=8, alpha=0.35, color="#2b8cbe", edgecolors="none")
        v_min = float(min(np.min(y_true), np.min(y_pred)))
        v_max = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([v_min, v_max], [v_min, v_max], "k--", linewidth=1.5, label="Paridade Ideal (1:1)")
        ax.set_title(f"Paridade Real vs. Predito — {model_name.upper()} ({split_name.upper()})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Real ({target_name})", fontsize=10)
        ax.set_ylabel(f"Predito ({target_name})", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p3 = model_dir / f"parity_{s_name}.{self.plot_format}"
        fig.savefig(p3, format=self.plot_format)
        plt.close(fig)
        saved["parity"] = str(p3)

        # 4. Erros Residuais
        residuals = y_pred - y_true
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        ax.plot(timestamps, residuals, color="#555555", linewidth=0.9, alpha=0.8)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(f"Resíduos (Predito - Real) — {model_name.upper()} ({split_name.upper()})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestamp", fontsize=10)
        ax.set_ylabel("Erro Residual", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p4 = model_dir / f"residuals_{s_name}.{self.plot_format}"
        fig.savefig(p4, format=self.plot_format)
        plt.close(fig)
        saved["residuals"] = str(p4)

        return saved

    def plot_models_comparison(self, comparison_df: pd.DataFrame) -> Dict[str, Path]:
        """
        Salva gráficos individuais em PDF para cada métrica comparativa:
        1. comparison_rmse.pdf: Comparativo de RMSE
        2. comparison_mae.pdf: Comparativo de MAE
        """
        if comparison_df.empty:
            return {}

        saved = {}
        for metric in ["RMSE", "MAE"]:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            models = sorted(comparison_df["model"].unique())
            splits = sorted(comparison_df["split"].unique())
            x = np.arange(len(models))
            width = 0.35 if len(splits) > 1 else 0.5

            for i, sp in enumerate(splits):
                sub = comparison_df[comparison_df["split"] == sp].set_index("model")
                vals = [sub.loc[m, metric] if m in sub.index else 0.0 for m in models]
                offset = (i - (len(splits) - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, label=sp.upper(), alpha=0.85)
                for b in bars:
                    h = b.get_height()
                    if h > 0:
                        ax.annotate(
                            f"{h:.4f}",
                            (b.get_x() + b.get_width() / 2, h),
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in models], fontsize=10, fontweight="bold")
            ax.set_title(f"Comparativo de Desempenho — {metric} (Escala Física Original)", fontsize=11, fontweight="bold")
            ax.set_ylabel(metric, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()

            out_path = self.base_output_dir / f"comparison_{metric.lower()}.{self.plot_format}"
            fig.savefig(out_path, format=self.plot_format)
            plt.close(fig)
            saved[metric.lower()] = out_path

        return saved
