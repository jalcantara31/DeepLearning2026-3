"""
Pré-processamento estritamente livre de Data Leakage para séries temporais multivariadas.

Garantias de engenharia contra vazamento de dados (Data Leakage):
1. A divisão (Treino / Validação / Teste) é estritamente cronológica e contígua,
   realizada ANTES de qualquer imputação, normalização ou criação de janelas.
2. O preenchimento de micro-lacunas usa apenas forward-fill causal (ffill)
   isoladamente dentro de cada partição (jamais bfill ou interpolação bilateral que olhe t+1).
3. Os escalonadores (feature_scaler e target_scaler) executam .fit() EXCLUSIVAMENTE
   sobre as linhas válidas do conjunto de TREINO.
4. Uma rotina de auditoria em tempo de execução (verify_no_leakage) valida matematicamente
   todas as invariantes antes do treinamento.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ai.loader.data_loader import CompressorDataLoader

logger = logging.getLogger(__name__)


@dataclass
class PreparedSplits:
    """Estrutura de saída contendo as partições escalonadas, brutas e metadados do pré-processamento."""

    train_scaled: pd.DataFrame
    val_scaled: pd.DataFrame
    test_scaled: Optional[pd.DataFrame]
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    test_raw: Optional[pd.DataFrame]
    feature_cols: List[str]
    target_cols: List[str]
    feature_scaler: StandardScaler | RobustScaler | MinMaxScaler
    target_scaler: StandardScaler | RobustScaler | MinMaxScaler
    leakage_audit: Dict[str, object]


class TimeSeriesPreprocessor:
    """
    Pipeline modular de pré-processamento causal e escalonamento sem data leakage.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_cols: str | List[str] = "act_poly_eff",
        resample_freq: Optional[str] = "5min",
        filter_physical_anomalies: bool = True,
        min_power_kw: float = 4000.0,
        min_press_ratio: float = 1.05,
        min_eff_pct: float = 50.0,
        max_eff_pct: float = 100.0,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_gap_steps: int = 1,
        max_ffill_steps: int = 3,
        scaler_type: str = "standard",
    ) -> None:
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        self.target_cols = [CompressorDataLoader.normalize_col_name(c) for c in target_cols]
        self.feature_cols = (
            [CompressorDataLoader.normalize_col_name(c) for c in feature_cols]
            if feature_cols is not None
            else None
        )
        self.resample_freq = resample_freq
        self.filter_physical_anomalies = filter_physical_anomalies
        self.min_power_kw = float(min_power_kw)
        self.min_press_ratio = float(min_press_ratio)
        self.min_eff_pct = float(min_eff_pct)
        self.max_eff_pct = float(max_eff_pct)

        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-3):
            raise ValueError(
                f"A soma de train_ratio ({train_ratio}) + val_ratio ({val_ratio}) + "
                f"test_ratio ({test_ratio}) deve ser 1.0 (recebido: {total_ratio:.4f})."
            )
        if train_ratio <= 0 or val_ratio <= 0:
            raise ValueError("Tanto train_ratio quanto val_ratio devem ser estritamente positivos.")

        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_gap_steps = max(0, int(split_gap_steps))
        self.max_ffill_steps = max(0, int(max_ffill_steps))
        self.scaler_type = scaler_type.lower().strip()

    def _create_scaler(self) -> StandardScaler | RobustScaler | MinMaxScaler:
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        raise ValueError(
            f"Tipo de scaler inválido: '{self.scaler_type}'. Opções válidas: 'standard', 'robust', 'minmax'."
        )

    def _apply_instantaneous_physical_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mascara paradas operacionais e glitches termodinâmicos usando regras puramente
        instantâneas (apenas na linha t), sem utilizar estatísticas globais da série.
        """
        if not self.filter_physical_anomalies:
            return df

        invalid_mask = pd.Series(False, index=df.index)
        if "act_shaft_power" in df.columns:
            invalid_mask |= df["act_shaft_power"] <= self.min_power_kw
        if "act_press_ratio" in df.columns:
            invalid_mask |= df["act_press_ratio"] < self.min_press_ratio
        if "act_poly_eff" in df.columns:
            invalid_mask |= (df["act_poly_eff"] < self.min_eff_pct) | (
                df["act_poly_eff"] > self.max_eff_pct
            )

        n_masked = int(invalid_mask.sum())
        logger.info(
            "Filtro físico instantâneo (paradas/glitches): %d instantes marcados como NaN (%.2f%%).",
            n_masked,
            n_masked / max(1, len(df)) * 100.0,
        )
        return df.mask(invalid_mask)

    def _resolve_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        for t_col in self.target_cols:
            if t_col not in df.columns:
                raise KeyError(
                    f"Coluna alvo '{t_col}' não encontrada no DataFrame. Colunas disponíveis: {list(df.columns)}"
                )

        if self.feature_cols is None:
            # Por padrão, utiliza as variáveis físicas principais (Actual e Expected)
            default_candidates = [
                "act_press_ratio",
                "exp_press_ratio",
                "act_poly_head",
                "exp_poly_head",
                "act_shaft_power",
                "exp_shaft_power",
                "act_poly_eff",
                "exp_poly_eff",
            ]
            feat_cols = [c for c in default_candidates if c in df.columns]
            for t_col in self.target_cols:
                if t_col not in feat_cols:
                    feat_cols.append(t_col)
        else:
            missing_feats = [c for c in self.feature_cols if c not in df.columns]
            if missing_feats:
                raise KeyError(
                    f"Colunas de features não encontradas: {missing_feats}. Disponíveis: {list(df.columns)}"
                )
            feat_cols = list(self.feature_cols)

        return feat_cols, list(self.target_cols)

    @staticmethod
    def _transform_preserving_nan(
        df_split: pd.DataFrame,
        cols: List[str],
        scaler: StandardScaler | RobustScaler | MinMaxScaler,
    ) -> pd.DataFrame:
        """
        Aplica scaler.transform apenas nas linhas completas (sem NaN),
        preservando NaN nas linhas de parada/lacuna para descarte seguro no janelamento.
        """
        out = pd.DataFrame(np.nan, index=df_split.index, columns=cols, dtype=np.float32)
        valid_mask = df_split[cols].notna().all(axis=1)
        if valid_mask.any():
            transformed = scaler.transform(df_split.loc[valid_mask, cols].values)
            out.loc[valid_mask, cols] = transformed.astype(np.float32)
        return out

    def process(self, df_raw: pd.DataFrame) -> PreparedSplits:
        """
        Executa a regularização temporal, filtragem física, divisão cronológica estrita,
        imputação causal isolada por split e escalonamento ajustado exclusivamente no treino.
        """
        df = df_raw.copy()

        # 1. Regularização temporal causal (closed='left', label='left' evita olhar o próximo bucket)
        if self.resample_freq:
            logger.info("Reamostrando série na grade regular de '%s'...", self.resample_freq)
            df = df.resample(self.resample_freq, closed="left", label="left").mean()

        # 2. Mascaramento físico instantâneo (paradas e glitches de partida)
        df = self._apply_instantaneous_physical_mask(df)

        feat_cols, target_cols = self._resolve_columns(df)
        all_needed_cols = list(dict.fromkeys(feat_cols + target_cols))
        df = df[all_needed_cols]

        # 3. Divisão estritamente cronológica ANTES de qualquer imputação ou scaling
        n_total = len(df)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        gap = self.split_gap_steps

        train_end = n_train
        val_start = min(n_total, train_end + gap)
        val_end = min(n_total, val_start + n_val)
        test_start = min(n_total, val_end + gap)

        train_raw = df.iloc[:train_end].copy()
        val_raw = df.iloc[val_start:val_end].copy()
        test_raw = df.iloc[test_start:].copy() if self.test_ratio > 0 and test_start < n_total else None

        # 4. Preenchimento causal (SOMENTE forward-fill curto) isolado dentro de cada partição
        if self.max_ffill_steps > 0:
            train_raw = train_raw.ffill(limit=self.max_ffill_steps)
            val_raw = val_raw.ffill(limit=self.max_ffill_steps)
            if test_raw is not None:
                test_raw = test_raw.ffill(limit=self.max_ffill_steps)

        # 5. Ajuste (FIT) dos escalonadores EXCLUSIVAMENTE nos dados válidos de TREINO
        train_valid_mask = train_raw[all_needed_cols].notna().all(axis=1)
        n_train_valid = int(train_valid_mask.sum())
        if n_train_valid == 0:
            raise ValueError("Nenhuma linha válida encontrada no conjunto de treino após pré-processamento.")

        feature_scaler = self._create_scaler()
        target_scaler = self._create_scaler()

        feature_scaler.fit(train_raw.loc[train_valid_mask, feat_cols].values)
        target_scaler.fit(train_raw.loc[train_valid_mask, target_cols].values)

        # 6. Transformação isolada de Treino, Validação e Teste usando os parâmetros congelados do Treino
        train_scaled = self._transform_preserving_nan(train_raw, feat_cols, feature_scaler)
        val_scaled = self._transform_preserving_nan(val_raw, feat_cols, feature_scaler)
        test_scaled = (
            self._transform_preserving_nan(test_raw, feat_cols, feature_scaler)
            if test_raw is not None and len(test_raw) > 0
            else None
        )

        # Se alguma coluna alvo não estiver em feat_cols, adiciona escalonada pelo target_scaler
        extra_targets = [c for c in target_cols if c not in feat_cols]
        if extra_targets:
            for split_raw, split_scaled in [
                (train_raw, train_scaled),
                (val_raw, val_scaled),
                (test_raw, test_scaled),
            ]:
                if split_raw is not None and split_scaled is not None:
                    t_trans = self._transform_preserving_nan(split_raw, target_cols, target_scaler)
                    for c in extra_targets:
                        split_scaled[c] = t_trans[c]

        # 7. Auditoria matemática formal contra Data Leakage
        leakage_audit = self.verify_no_leakage(
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
            feature_scaler=feature_scaler,
            n_train_valid=n_train_valid,
            feat_cols=feat_cols,
        )

        logger.info(
            "Partições prontas (ZERO Data Leakage verificado) | "
            "Treino: %d (%s -> %s) | Validação: %d (%s -> %s)%s",
            len(train_scaled),
            train_scaled.index.min(),
            train_scaled.index.max(),
            len(val_scaled),
            val_scaled.index.min(),
            val_scaled.index.max(),
            (
                f" | Teste: {len(test_scaled)} ({test_scaled.index.min()} -> {test_scaled.index.max()})"
                if test_scaled is not None
                else ""
            ),
        )

        return PreparedSplits(
            train_scaled=train_scaled,
            val_scaled=val_scaled,
            test_scaled=test_scaled,
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
            feature_cols=feat_cols,
            target_cols=target_cols,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            leakage_audit=leakage_audit,
        )

    @staticmethod
    def verify_no_leakage(
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_raw: Optional[pd.DataFrame],
        feature_scaler: StandardScaler | RobustScaler | MinMaxScaler,
        n_train_valid: int,
        feat_cols: List[str],
    ) -> Dict[str, object]:
        """
        Verifica matematicamente que não há qualquer sobreposição temporal nem
        contaminação estatística entre Treino, Validação e Teste.
        Levanta AssertionError caso qualquer invariante seja violada.
        """
        assert train_raw.index.is_monotonic_increasing, "Índice de treino não está ordenado!"
        assert val_raw.index.is_monotonic_increasing, "Índice de validação não está ordenado!"

        train_max_ts = train_raw.index.max()
        val_min_ts = val_raw.index.min()
        val_max_ts = val_raw.index.max()

        assert train_max_ts < val_min_ts, (
            f"VIOLAÇÃO DE DATA LEAKAGE: Treino termina em {train_max_ts}, "
            f"mas Validação começa em {val_min_ts}!"
        )
        assert len(train_raw.index.intersection(val_raw.index)) == 0, (
            "VIOLAÇÃO DE DATA LEAKAGE: Interseção de timestamps entre Treino e Validação!"
        )

        test_min_ts = None
        test_max_ts = None
        if test_raw is not None and len(test_raw) > 0:
            assert test_raw.index.is_monotonic_increasing, "Índice de teste não está ordenado!"
            test_min_ts = test_raw.index.min()
            test_max_ts = test_raw.index.max()
            assert val_max_ts < test_min_ts, (
                f"VIOLAÇÃO DE DATA LEAKAGE: Validação termina em {val_max_ts}, "
                f"mas Teste começa em {test_min_ts}!"
            )
            assert len(train_raw.index.intersection(test_raw.index)) == 0
            assert len(val_raw.index.intersection(test_raw.index)) == 0

        # Verifica que o scaler viu exatamente n_train_valid amostras (quando aplicável)
        if hasattr(feature_scaler, "n_samples_seen_"):
            seen = int(np.max(feature_scaler.n_samples_seen_))
            assert seen == n_train_valid, (
                f"VIOLAÇÃO DE DATA LEAKAGE: Scaler viu {seen} amostras, "
                f"mas o treino válido possui {n_train_valid} amostras!"
            )

        # Verifica que a média do StandardScaler bate exatamente com a média apenas do treino
        if isinstance(feature_scaler, StandardScaler):
            expected_mean = train_raw[feat_cols].dropna().mean().values
            assert np.allclose(feature_scaler.mean_, expected_mean, rtol=1e-5, atol=1e-6), (
                "VIOLAÇÃO DE DATA LEAKAGE: Média do StandardScaler diverge da média exclusiva do treino!"
            )

        return {
            "leakage_detected": False,
            "chronological_order_verified": True,
            "zero_timestamp_overlap_verified": True,
            "scaler_fitted_exclusively_on_train": True,
            "train_Valid_samples_fitted": n_train_valid,
            "train_period": [str(train_raw.index.min()), str(train_max_ts)],
            "val_period": [str(val_min_ts), str(val_max_ts)],
            "test_period": [str(test_min_ts), str(test_max_ts)] if test_min_ts is not None else None,
        }

