"""Motor de treinamento e validação em PyTorch puro (sem PyTorch Lightning)."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Histórico época a época do treinamento e validação."""

    model_name: str
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    epoch_time_sec: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    checkpoint_path: Optional[str] = None


class PyTorchTrainer:
    """
    Treinador modular em PyTorch puro com suporte a:
    - Funções de perda: MSE, Huber (SmoothL1) e MAE (L1)
    - Otimizadores: AdamW, Adam e SGD
    - Escalonadores de taxa de aprendizado: ReduceLROnPlateau e CosineAnnealingLR
    - Clipping de norma de gradiente (evita explosão de gradiente em RNNs)
    - Early Stopping monitorando exclusivamente a perda de validação (val_loss)
    - Salvamento seguro de pesos (state_dict puro compatível com weights_only=True)
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn_name: str = "huber",
        optimizer_name: str = "adamw",
        scheduler_name: str = "reduce_on_plateau",
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 8,
        early_stopping_min_delta: float = 1e-5,
        checkpoint_dir: Optional[str | Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = float(grad_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)

        self.criterion = self._build_criterion(loss_fn_name)
        self.optimizer = self._build_optimizer(optimizer_name)
        self.scheduler_name = scheduler_name.lower().strip()
        self.scheduler = self._build_scheduler(self.scheduler_name)

        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_criterion(name: str) -> nn.Module:
        key = name.lower().strip()
        if key in ("mse", "l2"):
            return nn.MSELoss()
        if key in ("huber", "smooth_l1"):
            return nn.HuberLoss(delta=1.0)
        if key in ("mae", "l1"):
            return nn.L1Loss()
        raise ValueError(f"Função de perda '{name}' inválida. Use 'mse', 'huber' ou 'mae'.")

    def _build_optimizer(self, name: str) -> torch.optim.Optimizer:
        key = name.lower().strip()
        if key == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if key == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if key == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        raise ValueError(f"Otimizador '{name}' inválido. Use 'adamw', 'adam' ou 'sgd'.")

    def _build_scheduler(self, name: str):
        if name in ("none", ""):
            return None
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.epochs), eta_min=1e-6
            )
        raise ValueError(
            f"Scheduler '{name}' inválido. Use 'reduce_on_plateau', 'cosine' ou 'none'."
        )

    def _run_train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()

            if self.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            self.optimizer.step()

            batch_sz = x_batch.size(0)
            total_loss += float(loss.item()) * batch_sz
            total_samples += batch_sz

        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate_loss(self, val_loader: DataLoader) -> float:
        """Avalia a perda média em modo de inferência (sem atualizar gradientes ou estatísticas)."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)

            batch_sz = x_batch.size(0)
            total_loss += float(loss.item()) * batch_sz
            total_samples += batch_sz

        return total_loss / max(1, total_samples)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        """
        Executa o ciclo completo de treinamento com validação a cada época e Early Stopping.
        Restaura automaticamente os pesos da melhor época de validação ao final.
        """
        history = TrainingHistory(model_name=self.model_name)
        best_state_dict = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        ckpt_path = (
            self.checkpoint_dir / f"best_{self.model_name.lower()}.pt"
            if self.checkpoint_dir is not None
            else None
        )

        logger.info(
            "Iniciando treinamento [%s] em %s | Parâmetros: %s | Épocas máx: %d",
            self.model_name.upper(),
            self.device,
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}",
            self.epochs,
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._run_train_epoch(train_loader)
            val_loss = self.evaluate_loss(val_loader)
            elapsed = time.perf_counter() - t0

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.lr.append(current_lr)
            history.epoch_time_sec.append(elapsed)

            if self.scheduler is not None:
                if self.scheduler_name == "reduce_on_plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            improved = val_loss < (history.best_val_loss - self.early_stopping_min_delta)
            if improved:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if ckpt_path is not None:
                    torch.save(best_state_dict, ckpt_path)
                    history.checkpoint_path = str(ckpt_path)
                marker = "* [MELHOR]"
            else:
                patience_counter += 1
                marker = f"(paciência {patience_counter}/{self.early_stopping_patience})"

            logger.info(
                "[%s] Época %03d/%03d | train_loss: %.6f | val_loss: %.6f | lr: %.2e | %.2fs %s",
                self.model_name.upper(),
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                current_lr,
                elapsed,
                marker,
            )

            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    "[%s] Early stopping acionado na época %d (melhor época: %d, val_loss=%.6f).",
                    self.model_name.upper(),
                    epoch,
                    history.best_epoch,
                    history.best_val_loss,
                )
                break

        # Restaura os melhores pesos de validação
        self.model.load_state_dict(best_state_dict)
        return history

