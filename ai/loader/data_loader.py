"""Carregador modular e seguro para os dados multivariados do compressor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapeamento oficial entre nomes originais do historiador PI/CBM e nomes padronizados
COL_MAP: Dict[str, str] = {
    "PH (CBM) 1st Stage ExpPress Ratio": "exp_press_ratio",
    "PH (CBM) 1st Stage Poly Head Dev": "poly_head_dev",
    "PH (CBM) 1st Stage ActShaft Power": "act_shaft_power",
    "PH (CBM) 1st Stage Press Rat Dev": "press_rat_dev",
    "PH (CBM) 1st Stg ActCompr Poly Head": "act_poly_head",
    "PH (CBM) 1st Stage Poly Eff Dev": "poly_eff_dev_raw",
    "PH (CBM) 1st Stage Shft Pwr Dev": "shaft_pwr_dev",
    "PH (CBM) 1st Stage ActCompr Poly Eff": "act_poly_eff",
    "PH (CBM) 1st Stage ExpCompr Poly Eff": "exp_poly_eff",
    "PH (CBM) 1st Stage ExpShaft Power": "exp_shaft_power",
    "PH (CBM) 1st Stage ActPress Ratio": "act_press_ratio",
    "PH (CBM) 1st Stg ExpCompr Poly Head": "exp_poly_head",
}
INV_COL_MAP: Dict[str, str] = {v: k for k, v in COL_MAP.items()}


class CompressorDataLoader:
    """
    Responsável pela leitura segura, validação de schema, conversão temporal
    e padronização das colunas de sensores do compressor.
    """

    def __init__(
        self,
        csv_path: str | Path,
        reconstruct_poly_eff_dev: bool = True,
        drop_all_nan_cols: bool = True,
    ) -> None:
        self.csv_path = self._resolve_csv_path(csv_path)
        self.reconstruct_poly_eff_dev = reconstruct_poly_eff_dev
        self.drop_all_nan_cols = drop_all_nan_cols

    @staticmethod
    def _resolve_csv_path(csv_path: str | Path) -> Path:
        raw_path = Path(csv_path)
        repo_root = Path(__file__).resolve().parents[2]
        candidates: List[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path.resolve())
        else:
            candidates.extend([
                (Path.cwd() / raw_path).resolve(),
                (repo_root / raw_path).resolve(),
                (repo_root / "data" / raw_path.name).resolve(),
            ])

        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() == ".csv":
                return candidate

        raise FileNotFoundError(
            f"Arquivo CSV não encontrado para '{csv_path}'. "
            f"Caminhos verificados: {[str(c) for c in candidates]}"
        )

    @staticmethod
    def normalize_col_name(col_name: str) -> str:
        """Converte nome original da tag ou nome curto para o identificador padronizado."""
        return COL_MAP.get(col_name, col_name)

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Carrega o CSV, converte o índice para DatetimeIndex estritamente ordenado
        e aplica a padronização de nomes de colunas sem qualquer operação entre instantes.
        """
        logger.info("Carregando dataset a partir de: %s", self.csv_path)
        df = pd.read_csv(self.csv_path, index_col=0, nrows=nrows)

        # Conversão e ordenação estritamente cronológica do índice
        df.index = pd.to_datetime(df.index, format="ISO8601")
        df.index.name = "timestamp"
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Caso existam timestamps idênticos, consolida apenas no mesmo instante t
        if df.index.duplicated().any():
            n_dups = int(df.index.duplicated().sum())
            logger.warning("Encontrados %d timestamps duplicados; consolidando por média instantânea.", n_dups)
            df = df.groupby(level=0).mean()

        df = df.rename(columns=COL_MAP)

        # Remove colunas 100% vazias (ex.: poly_eff_dev_raw)
        if self.drop_all_nan_cols:
            empty_cols = [c for c in df.columns if df[c].isna().all()]
            if empty_cols:
                logger.info("Removendo colunas 100%% nulas: %s", empty_cols)
                df = df.drop(columns=empty_cols)

        # Reconstrução puramente instantânea (no mesmo instante t, sem olhar passado/futuro)
        if (
            self.reconstruct_poly_eff_dev
            and "act_poly_eff" in df.columns
            and "exp_poly_eff" in df.columns
            and "poly_eff_dev" not in df.columns
        ):
            exp_safe = df["exp_poly_eff"].where(df["exp_poly_eff"].abs() > 1.0, np.nan)
            df["poly_eff_dev"] = (df["act_poly_eff"] - exp_safe) / exp_safe * 100.0
            df.loc[df["exp_poly_eff"] == 0.0, "poly_eff_dev"] = 0.0

        logger.info(
            "Dataset carregado: %d linhas x %d colunas | Período: %s -> %s",
            df.shape[0],
            df.shape[1],
            df.index.min(),
            df.index.max(),
        )
        return df

