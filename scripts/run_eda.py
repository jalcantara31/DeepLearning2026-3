#!/usr/bin/env python3
"""
Script autônomo e modular para execução completa da Análise Exploratória de Dados (EDA)
e exportação individual de gráficos em formato PDF vetorial em results/eda/.

Baseado em: eda/analise_compressor.ipynb
Dados: data/compressor.csv

Uso:
    python scripts/run_eda.py
    python scripts/run_eda.py --data-path data/compressor.csv --output-dir results/eda --format pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend não interativo para geração rápida de arquivos
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de estilo visual consistente
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 120,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COL_MAP = {
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

LABELS_PT = {
    "act_press_ratio": "Razão Pressão Real (-)",
    "exp_press_ratio": "Razão Pressão Esperada (-)",
    "press_rat_dev": "Desvio Razão Pressão (%)",
    "act_poly_head": "Head Politrópico Real (kJ/kg)",
    "exp_poly_head": "Head Politrópico Esperado (kJ/kg)",
    "poly_head_dev": "Desvio Head Politrópico (%)",
    "act_shaft_power": "Potência de Eixo Real (kW)",
    "exp_shaft_power": "Potência de Eixo Esperada (kW)",
    "shaft_pwr_dev": "Desvio Potência de Eixo (%)",
    "act_poly_eff": "Eficiência Politrópica Real (%)",
    "exp_poly_eff": "Eficiência Politrópica Esperada (%)",
    "poly_eff_dev_raw": "Desvio Eficiência (Bruto) (%)",
    "poly_eff_dev": "Desvio Eficiência Reconstruído (%)",
}

SENSOR_COLS = [
    "act_press_ratio", "exp_press_ratio", "press_rat_dev",
    "act_poly_head", "exp_poly_head", "poly_head_dev",
    "act_shaft_power", "exp_shaft_power", "shaft_pwr_dev",
    "act_poly_eff", "exp_poly_eff", "poly_eff_dev",
]


class CompressorEDAPipeline:
    """Pipeline para execução de todas as análises exploratórias e salvamento de plots."""

    def __init__(self, data_path: Path, output_dir: Path, plot_format: str = "pdf") -> None:
        self.data_path = Path(data_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.series_dir = self.output_dir / "series_temporais"
        self.series_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format.lower().lstrip(".")
        self.logger = logging.getLogger("CompressorEDA")

    def _save_fig(self, fig: plt.Figure, filename: str, subfolder: Path | None = None) -> None:
        target_dir = subfolder if subfolder is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(f".{self.plot_format}"):
            filename = f"{filename}.{self.plot_format}"
        out_path = target_dir / filename
        fig.savefig(out_path, format=self.plot_format, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("  [Salvo] %s", str(out_path.relative_to(self.output_dir.parent)))

    def load_and_clean_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Carrega dados brutos, diagnostica e cria grade de 30s e estados operacionais."""
        self.logger.info("1/8. Carregando e estruturando dados de %s...", self.data_path)
        df_raw = pd.read_csv(self.data_path, index_col=0)
        df_raw.index = pd.to_datetime(df_raw.index, format="ISO8601")
        df_raw.index.name = "timestamp"
        df_raw = df_raw.sort_index()
        df = df_raw.rename(columns=COL_MAP)

        # Diagnóstico de taxa de amostragem
        dt_seconds = df.index.to_series().diff().dt.total_seconds()
        top_dts = dt_seconds.value_counts().head(6)

        # Plot 01: Delta t
        fig1, ax1 = plt.subplots(figsize=(8.5, 4.5))
        sns.barplot(
            x=[f"{int(k)}s" for k in top_dts.index],
            y=top_dts.values,
            ax=ax1,
            palette="Blues_r",
        )
        ax1.set_yscale("log")
        ax1.set_title(r"Distribuição dos Principais Intervalos $\Delta t$ (Escala Log)")
        ax1.set_xlabel("Intervalo entre registros consecutivos")
        ax1.set_ylabel("Contagem de registros (log)")
        for p in ax1.patches:
            h = p.get_height()
            if h > 0:
                ax1.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2, h),
                             ha="center", va="bottom", fontsize=8)
        self._save_fig(fig1, "01_diagnostico_amostragem_delta_t")

        # Plot 02: Valores Ausentes
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
        miss_pct = (df.isna().mean() * 100).sort_values(ascending=True)
        miss_pct.plot(kind="barh", ax=ax2, color="#d95f02")
        ax2.set_title("Percentual de Valores Ausentes por Variável (Dados Brutos)")
        ax2.set_xlabel("% de Valores Ausentes")
        ax2.axvline(5.24, color="navy", linestyle="--", linewidth=1.2, label="~5.24% (Split-Row 2s)")
        ax2.legend()
        self._save_fig(fig2, "02_percentual_valores_ausentes")

        # Reamostragem na grade de 30s
        active_cols_11 = [c for c in df.columns if c != "poly_eff_dev_raw"]
        df_30s = df[active_cols_11].resample("30s").mean()

        # Reconstrução física de poly_eff_dev
        exp_eff_safe = df_30s["exp_poly_eff"].where(df_30s["exp_poly_eff"] > 1.0, np.nan)
        df_30s["poly_eff_dev"] = (df_30s["act_poly_eff"] - exp_eff_safe) / exp_eff_safe * 100.0
        df_30s.loc[df_30s["exp_poly_eff"] == 0.0, "poly_eff_dev"] = 0.0
        df_30s = df_30s[SENSOR_COLS]

        # Classificação de estados operacionais
        state = pd.Series("Operação Normal", index=df_30s.index, name="estado_operacional")
        is_missing = df_30s[["act_shaft_power", "act_press_ratio", "act_poly_eff"]].isna().any(axis=1)
        is_offline = (~is_missing) & ((df_30s["act_shaft_power"] <= 500.0) | (df_30s["act_press_ratio"] < 1.05))
        is_glitch = (~is_missing) & (~is_offline) & (
            (df_30s["act_poly_eff"] < 50.0)
            | (df_30s["act_poly_eff"] > 100.0)
            | (df_30s["act_shaft_power"] < 4000.0)
        )
        state.loc[is_glitch] = "Transiente / Glitch"
        state.loc[is_offline] = "Parada (Offline)"
        state.loc[is_missing] = "Gap / Sem Leitura"

        return df, df_30s, state

    def plot_macro_and_micro(self, df_30s: pd.DataFrame, state: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gera séries temporais macro de 6 meses e zoom micro em transientes."""
        self.logger.info("2/8. Gerando séries temporais macro e zoom de parada...")
        df_norm_only = df_30s.where(state == "Operação Normal")
        df_clean_30s = df_norm_only.interpolate(method="time", limit=4)
        df_5min = df_norm_only.resample("5min").mean()
        df_1h = df_norm_only.resample("1h").mean()
        df_1d = df_norm_only.resample("1D").mean()

        families_plot = [
            ("razao_pressao", "Razão de Compressão (-)", "act_press_ratio", "exp_press_ratio", "#1f77b4", "#ff7f0e"),
            ("head_politropico", "Head Politrópico (kJ/kg)", "act_poly_head", "exp_poly_head", "#2ca02c", "#d62728"),
            ("potencia_eixo", "Potência de Eixo (kW)", "act_shaft_power", "exp_shaft_power", "#9467bd", "#8c564b"),
            ("eficiencia_politropica", "Eficiência Politrópica (%)", "act_poly_eff", "exp_poly_eff", "#008080", "#e377c2"),
        ]

        shutdown_dates = [
            pd.Timestamp("2019-09-12 18:40"),
            pd.Timestamp("2019-09-24 05:10"),
            pd.Timestamp("2019-10-10 10:40"),
        ]

        # Plots 03: Séries Macro das 4 famílias
        for tag, title, act_c, exp_c, c_act, c_exp in families_plot:
            fig, ax = plt.subplots(figsize=(12, 4.2))
            ax.plot(df_1h.index, df_1h[act_c], color=c_act, alpha=0.30, linewidth=0.8, label="Actual (Média 1h)")
            ax.plot(df_1d.index, df_1d[act_c], color=c_act, linewidth=2.0, label="Actual (Média Diária)")
            ax.plot(df_1d.index, df_1d[exp_c], color=c_exp, linewidth=1.8, linestyle="--", label="Expected (Média Diária)")
            for i_sd, sd_ts in enumerate(shutdown_dates):
                ax.axvline(sd_ts, color="crimson", linestyle=":", alpha=0.75,
                           label="Parada Operacional" if i_sd == 0 else None)
            ax.set_title(f"{title} — Comparação Actual vs. Expected (Set/2019 a Mar/2020)")
            ax.set_ylabel(title.split("(")[-1].rstrip(")"))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.set_xlabel("Data")
            ax.legend(loc="upper right", ncol=4)
            self._save_fig(fig, f"03_serie_macro_{tag}")

        # Plots Individuais de CADA UMA das 12 Séries Temporais na pasta results/eda/series_temporais/
        sensor_colors = {
            "act_press_ratio": "#1f77b4", "exp_press_ratio": "#ff7f0e", "press_rat_dev": "#1b9e77",
            "act_poly_head": "#2ca02c", "exp_poly_head": "#d62728", "poly_head_dev": "#d95f02",
            "act_shaft_power": "#9467bd", "exp_shaft_power": "#8c564b", "shaft_pwr_dev": "#7570b3",
            "act_poly_eff": "#008080", "exp_poly_eff": "#e377c2", "poly_eff_dev": "#e7298a",
        }
        for col in SENSOR_COLS:
            fig_s, ax_s = plt.subplots(figsize=(12, 4.2))
            cor = sensor_colors.get(col, "#2b8cbe")
            ax_s.plot(df_1h.index, df_1h[col], color=cor, alpha=0.35, linewidth=0.8, label="Média Horária (1h)")
            ax_s.plot(df_1d.index, df_1d[col], color=cor, linewidth=2.0, label="Tendência Diária (1D)")
            for i_sd, sd_ts in enumerate(shutdown_dates):
                ax_s.axvline(sd_ts, color="crimson", linestyle=":", alpha=0.75,
                             label="Parada Operacional" if i_sd == 0 else None)
            ax_s.set_title(f"Série Temporal — {LABELS_PT[col]} (Set/2019 a Mar/2020)")
            ax_s.set_ylabel(LABELS_PT[col])
            ax_s.set_xlabel("Data")
            ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax_s.legend(loc="upper right")
            self._save_fig(fig_s, f"serie_{col}", subfolder=self.series_dir)

        # Plots 04 e 05: Desvios CBM
        dev_cols = ["press_rat_dev", "poly_head_dev", "shaft_pwr_dev", "poly_eff_dev"]
        dev_rolling_24h = df_1h[dev_cols].rolling(window=24, min_periods=6).mean()
        dev_rolling_7d = df_1h[dev_cols].rolling(window=24 * 7, min_periods=24).mean()
        palette_dev = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

        fig4, ax4 = plt.subplots(figsize=(12, 4.5))
        for col, cor in zip(dev_cols, palette_dev):
            ax4.plot(dev_rolling_24h.index, dev_rolling_24h[col], label=LABELS_PT[col], color=cor, linewidth=1.3, alpha=0.85)
        ax4.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax4.set_title("Evolução dos 4 Desvios CBM — Média Móvel de 24 Horas")
        ax4.set_ylabel("Desvio Relativo (%)")
        ax4.set_xlabel("Data")
        ax4.legend(loc="upper right", ncol=2)
        self._save_fig(fig4, "04_desvios_cbm_media_movel_24h")

        fig5, ax5 = plt.subplots(figsize=(12, 4.5))
        for col, cor in zip(dev_cols, palette_dev):
            ax5.plot(dev_rolling_7d.index, dev_rolling_7d[col], label=LABELS_PT[col], color=cor, linewidth=2.2)
        ax5.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax5.axvspan(pd.Timestamp("2019-09-01"), pd.Timestamp("2019-11-10"), color="#fee0d2", alpha=0.28, label="Ciclo 1: Queda gradual")
        ax5.axvspan(pd.Timestamp("2019-11-20"), pd.Timestamp("2019-12-20"), color="#e5f5e0", alpha=0.35, label="Recuperação")
        ax5.axvspan(pd.Timestamp("2019-12-20"), pd.Timestamp("2020-03-01"), color="#deebf7", alpha=0.30, label="Ciclo 2: Nova deriva")
        ax5.set_title("Tendência Estrutural dos Desvios CBM — Média Móvel de 7 Dias")
        ax5.set_ylabel("Desvio Relativo (%)")
        ax5.set_xlabel("Data")
        ax5.legend(loc="upper right", ncol=2)
        self._save_fig(fig5, "05_desvios_cbm_tendencia_7d")

        # Plots 06 a 08: Zoom da Parada de 24-25/Set/2019
        zoom_start, zoom_end = "2019-09-24 03:00:00", "2019-09-25 10:00:00"
        df_zoom = df_30s.loc[zoom_start:zoom_end]

        fig6, ax6 = plt.subplots(figsize=(11, 4.0))
        ax6.plot(df_zoom.index, df_zoom["act_shaft_power"], marker=".", markersize=2.5, linewidth=1.0, color="#9467bd", label="Potência Real (kW)")
        ax6.plot(df_zoom.index, df_zoom["exp_shaft_power"], linewidth=1.0, linestyle="--", color="#8c564b", label="Potência Esperada (kW)")
        ax6.set_title("Zoom (30s) na Parada de 24–25/Set/2019 — Potência de Eixo (Compressão Deadband)")
        ax6.set_ylabel("Potência (kW)")
        ax6.legend(loc="upper right")
        self._save_fig(fig6, "06_zoom_parada_potencia_eixo")

        fig7, ax7 = plt.subplots(figsize=(11, 3.8))
        ax7.plot(df_zoom.index, df_zoom["act_press_ratio"], marker=".", markersize=2.5, linewidth=1.0, color="#1f77b4", label="Razão Pressão Real")
        ax7.set_title("Zoom (30s) na Parada de 24–25/Set/2019 — Razão de Pressão Real")
        ax7.set_ylabel("Razão de Pressão")
        ax7.legend(loc="upper right")
        self._save_fig(fig7, "07_zoom_parada_razao_pressao")

        fig8, ax8 = plt.subplots(figsize=(11, 4.2))
        ax8.plot(df_zoom.index, df_zoom["act_poly_eff"], marker="o", markersize=3, linewidth=1.0, color="crimson", label="Eficiência Bruta (%) — Sem Filtro")
        ax8.set_ylim(-120, 260)
        ax8.axhspan(70, 95, color="seagreen", alpha=0.15, label="Faixa Física Normal (70%–95%)")
        ax8.set_title("Transiente de Partida (25/Set 07:09–07:15): Spike Artificial por Divisão por Potência ~0")
        ax8.set_ylabel("Eficiência (%) [truncada]")
        ax8.legend(loc="upper left")
        self._save_fig(fig8, "08_zoom_parada_glitch_eficiencia")

        return df_clean_30s, df_5min, df_1h, df_1d

    def plot_distributions_and_seasonality(self, df_norm_only: pd.DataFrame, df_1h: pd.DataFrame) -> None:
        """Gera histogramas, boxplots padronizados, sazonalidade e STL."""
        self.logger.info("3/8. Gerando distribuições univariadas e sazonalidade multi-escala...")
        df_norm_valid = df_norm_only.dropna()
        sample_dist = df_norm_valid.sample(n=min(40_000, len(df_norm_valid)), random_state=RANDOM_SEED)

        # Plots 09: Histogramas individuais das 12 tags
        for col in SENSOR_COLS:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            sns.histplot(sample_dist[col], bins=45, kde=True, ax=ax, color="#2b8cbe", edgecolor="none", stat="density")
            ax.axvline(sample_dist[col].mean(), color="crimson", linestyle="--", linewidth=1.2, label="Média")
            ax.axvline(sample_dist[col].median(), color="black", linestyle=":", linewidth=1.2, label="Mediana")
            ax.set_title(f"Distribuição — {LABELS_PT[col]}", fontsize=10.5)
            ax.set_xlabel(LABELS_PT[col])
            ax.set_ylabel("Densidade")
            ax.legend(fontsize=8.5)
            self._save_fig(fig, f"09_distribuicao_{col}")

        # Plot 10: Boxplots Padronizados
        z_sample = pd.DataFrame(
            StandardScaler().fit_transform(sample_dist),
            columns=[LABELS_PT[c] for c in SENSOR_COLS],
        )
        fig10, ax10 = plt.subplots(figsize=(12, 5.5))
        sns.boxplot(data=z_sample, orient="h", fliersize=1.5, palette="vlag", ax=ax10)
        ax10.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax10.set_xlim(-5.5, 5.5)
        ax10.set_title("Comparação das Distribuições Padronizadas (Z-Score) em Operação Normal")
        ax10.set_xlabel("Z-Score (Desvios Padrão em relação à Média)")
        self._save_fig(fig10, "10_boxplots_zscore_padronizados")

        # Plots 11 a 14: Sazonalidade
        df_seas = df_1h.dropna().copy()
        df_seas["hora"] = df_seas.index.hour
        df_seas["dia_semana"] = df_seas.index.dayofweek
        df_seas["ano_mes"] = df_seas.index.strftime("%Y-%m")
        dias_pt = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

        fig11, ax11 = plt.subplots(figsize=(9, 4.5))
        hourly_grp = df_seas.groupby("hora")[["act_shaft_power", "act_poly_eff"]].agg(["mean", "sem"])
        h_idx = hourly_grp.index
        p_mean, p_sem = hourly_grp[("act_shaft_power", "mean")], hourly_grp[("act_shaft_power", "sem")]
        e_mean, e_sem = hourly_grp[("act_poly_eff", "mean")], hourly_grp[("act_poly_eff", "sem")]
        ax11_r = ax11.twinx()
        ax11.plot(h_idx, p_mean, color="#9467bd", marker="o", linewidth=2, label="Potência Real (kW)")
        ax11.fill_between(h_idx, p_mean - 1.96 * p_sem, p_mean + 1.96 * p_sem, color="#9467bd", alpha=0.18)
        ax11_r.plot(h_idx, e_mean, color="#008080", marker="s", linewidth=2, linestyle="--", label="Eficiência Real (%)")
        ax11_r.fill_between(h_idx, e_mean - 1.96 * e_sem, e_mean + 1.96 * e_sem, color="#008080", alpha=0.15)
        ax11.set_title("Ciclo Diurno Médio (IC 95%): Potência vs. Eficiência por Hora")
        ax11.set_xlabel("Hora do Timestamp (0h–23h)")
        ax11.set_ylabel("Potência de Eixo (kW)", color="#9467bd")
        ax11_r.set_ylabel("Eficiência Politrópica (%)", color="#008080")
        ax11_r.grid(False)
        self._save_fig(fig11, "11_sazonalidade_ciclo_diurno")

        fig12, ax12 = plt.subplots(figsize=(8.5, 4.5))
        sns.boxplot(data=df_seas, x="dia_semana", y="act_poly_eff", ax=ax12, palette="crest", showfliers=False)
        ax12.set_xticks(range(7))
        ax12.set_xticklabels(dias_pt)
        ax12.set_title("Distribuição da Eficiência Politrópica por Dia da Semana")
        ax12.set_xlabel("Dia da Semana")
        ax12.set_ylabel("Eficiência Politrópica Real (%)")
        self._save_fig(fig12, "12_sazonalidade_dia_da_semana")

        fig13, ax13 = plt.subplots(figsize=(8.5, 4.5))
        sns.boxplot(data=df_seas, x="ano_mes", y="act_poly_eff", ax=ax13, palette="Spectral", showfliers=False)
        ax13.set_title("Evolução Mensal da Eficiência Politrópica Real")
        ax13.set_xlabel("Ano-Mês")
        ax13.set_ylabel("Eficiência Politrópica Real (%)")
        self._save_fig(fig13, "13_sazonalidade_evolucao_mensal")

        fig14, ax14 = plt.subplots(figsize=(10, 4.5))
        piv_mh = df_seas.pivot_table(index="ano_mes", columns="hora", values="act_poly_eff", aggfunc="mean")
        sns.heatmap(piv_mh, cmap="YlGnBu", ax=ax14, cbar_kws={"label": "Eficiência Média (%)"})
        ax14.set_title("Mapa de Calor Sazonal (Ano-Mês × Hora) — Eficiência Politrópica")
        ax14.set_xlabel("Hora do Dia")
        ax14.set_ylabel("Ano-Mês")
        self._save_fig(fig14, "14_sazonalidade_heatmap_mes_hora")

        # Plots 15 a 18: STL
        eff_1h_interp = df_1h["act_poly_eff"].interpolate(method="time").bfill().ffill()
        if HAS_STATSMODELS:
            stl_res = STL(eff_1h_interp, period=24, robust=True).fit()
            trend_c, seas_c, resid_c = stl_res.trend, stl_res.seasonal, stl_res.resid
        else:
            trend_c = eff_1h_interp.rolling(window=24 * 7, center=True, min_periods=24).mean().bfill().ffill()
            detr = eff_1h_interp - trend_c
            seas_c = detr.groupby(detr.index.hour).transform("mean")
            resid_c = detr - seas_c

        stl_list = [
            ("15_stl_serie_original", "Série Horária Observada — Eficiência Politrópica Real (%)", eff_1h_interp, "#008080"),
            ("16_stl_tendencia", "Componente de Tendência — Degradação e Deriva Estrutural", trend_c, "#d95f02"),
            ("17_stl_sazonalidade_diurna", "Componente Sazonal Diurna (Período = 24h)", seas_c, "#1f77b4"),
            ("18_stl_residuos", "Componente Residual — Ruído de Alta Frequência", resid_c, "#555555"),
        ]
        for name, title, s_data, color in stl_list:
            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.plot(s_data.index, s_data, color=color, linewidth=1.1)
            if "Residual" in title:
                ax.axhline(0, color="crimson", linestyle="--", linewidth=0.9)
            ax.set_title(title)
            ax.set_xlabel("Data")
            self._save_fig(fig, name)

    def plot_temporal_dynamics(self, df_clean_30s: pd.DataFrame, df_5min: pd.DataFrame, df_1h: pd.DataFrame) -> None:
        """Gera ACF, PACF, Lead-Lag e Espectro de Welch."""
        self.logger.info("4/8. Analisando dinâmica temporal (ACF, PACF, Lead-Lag e Welch)...")
        s_eff_1h = df_1h["act_poly_eff"].interpolate(method="time").dropna().values
        s_pwr_1h = df_1h["act_shaft_power"].interpolate(method="time").dropna().values

        def _acf_calc(x: np.ndarray, nlags: int) -> np.ndarray:
            xc = x - np.mean(x)
            denom = np.dot(xc, xc)
            return np.array([1.0] + [np.dot(xc[:-k], xc[k:]) / denom for k in range(1, nlags + 1)])

        # Plots individuais de Autocorrelação (ACF) para todas as 12 variáveis
        lags_h = np.arange(169)
        for col in SENSOR_COLS:
            s_c = df_1h[col].interpolate(method="time").dropna().values
            acf_c = acf(s_c, nlags=168, fft=True) if HAS_STATSMODELS else _acf_calc(s_c, 168)
            fig_acf, ax_acf = plt.subplots(figsize=(8.5, 4.2))
            ax_acf.plot(lags_h, acf_c, color="#1f77b4", linewidth=1.8, label="ACF")
            conf_c = 1.96 / np.sqrt(len(s_c))
            ax_acf.axhspan(-conf_c, conf_c, color="orange", alpha=0.2, label="IC 95%")
            for d in range(1, 8):
                ax_acf.axvline(d * 24, color="crimson", linestyle=":", alpha=0.45)
            ax_acf.set_title(f"Autocorrelação (ACF até 7 dias) — {LABELS_PT[col]}")
            ax_acf.set_xlabel("Defasagem (horas)")
            ax_acf.set_ylabel("Autocorrelação")
            ax_acf.legend(loc="upper right")
            self._save_fig(fig_acf, f"20_autocorrelacao_acf_{col}")

        sub_30s = df_clean_30s.loc["2019-12-01":"2019-12-08", "act_shaft_power"].dropna().values
        acf_pwr_30s = acf(sub_30s, nlags=120, fft=True) if HAS_STATSMODELS else _acf_calc(sub_30s, 120)
        acf_eff_1h = acf(s_eff_1h, nlags=168, fft=True) if HAS_STATSMODELS else _acf_calc(s_eff_1h, 168)
        acf_pwr_1h = acf(s_pwr_1h, nlags=168, fft=True) if HAS_STATSMODELS else _acf_calc(s_pwr_1h, 168)
        pacf_eff_1h = pacf(s_eff_1h, nlags=48, method="ywm") if HAS_STATSMODELS else _acf_calc(np.diff(s_eff_1h), 48)

        # Plot 19: ACF Curto Prazo
        fig19, ax19 = plt.subplots(figsize=(8.5, 4.2))
        ax19.plot(np.arange(len(acf_pwr_30s)) * 0.5, acf_pwr_30s, color="#9467bd", linewidth=2)
        ax19.axhline(1.96 / np.sqrt(len(sub_30s)), color="gray", linestyle="--", linewidth=0.9)
        ax19.set_title("ACF de Curto Prazo (Grade 30s: 0 a 60 min) — Potência de Eixo Real")
        ax19.set_xlabel("Defasagem (minutos)")
        ax19.set_ylabel("Autocorrelação")
        self._save_fig(fig19, "19_dinamica_acf_curto_prazo_30s")

        # Plot 20: ACF Horária
        fig20, ax20 = plt.subplots(figsize=(8.5, 4.2))
        ax20.plot(np.arange(len(acf_eff_1h)), acf_eff_1h, color="#008080", linewidth=1.8, label="Eficiência Real")
        ax20.plot(np.arange(len(acf_pwr_1h)), acf_pwr_1h, color="#9467bd", linewidth=1.5, alpha=0.8, label="Potência Real")
        for d in range(1, 8):
            ax20.axvline(d * 24, color="crimson", linestyle=":", alpha=0.45)
        ax20.set_title("ACF Horária até 7 Dias (168h) — Picos em Múltiplos de 24h")
        ax20.set_xlabel("Defasagem (horas)")
        ax20.set_ylabel("Autocorrelação")
        ax20.legend()
        self._save_fig(fig20, "20_dinamica_acf_medio_prazo_168h")

        # Plot 21: PACF
        fig21, ax21 = plt.subplots(figsize=(8.5, 4.2))
        ax21.stem(np.arange(len(pacf_eff_1h)), pacf_eff_1h, basefmt=" ")
        conf = 1.96 / np.sqrt(len(s_eff_1h))
        ax21.axhspan(-conf, conf, color="orange", alpha=0.2, label="IC 95%")
        ax21.set_title("PACF Horária (0 a 48h) — Eficiência Real")
        ax21.set_xlabel("Defasagem (horas)")
        ax21.set_ylabel("Autocorrelação Parcial")
        ax21.legend()
        self._save_fig(fig21, "21_dinamica_pacf_horaria")

        # Plot 22: Correlação Cruzada (Lead-Lag)
        diff_5m = df_5min[["act_shaft_power", "act_press_ratio", "act_poly_eff"]].interpolate(limit=3).diff().dropna()
        def _cross_corr(x, y, max_lag=24):
            xz = (x.values - x.mean()) / x.std()
            yz = (y.values - y.mean()) / y.std()
            lags = np.arange(-max_lag, max_lag + 1)
            cc = [np.dot(xz[:k], yz[-k:]) / len(xz) if k < 0 else (np.dot(xz[k:], yz[:-k]) / len(xz) if k > 0 else np.dot(xz, yz) / len(xz)) for k in lags]
            return lags, np.array(cc)

        lags_5m, cc_pwr_pr = _cross_corr(diff_5m["act_shaft_power"], diff_5m["act_press_ratio"])
        _, cc_pr_eff = _cross_corr(diff_5m["act_press_ratio"], diff_5m["act_poly_eff"])
        fig22, ax22 = plt.subplots(figsize=(8.5, 4.5))
        ax22.plot(lags_5m * 5, cc_pwr_pr, marker=".", label=r"$\Delta$ Potência $\leftrightarrow$ $\Delta$ Razão Pressão", color="#1f77b4")
        ax22.plot(lags_5m * 5, cc_pr_eff, marker=".", label=r"$\Delta$ Razão Pressão $\leftrightarrow$ $\Delta$ Eficiência", color="#008080")
        ax22.axvline(0, color="black", linestyle="--", linewidth=1)
        ax22.set_title("Correlação Cruzada (Lead-Lag) das Variações (5 min)")
        ax22.set_xlabel("Defasagem (minutos)")
        ax22.set_ylabel("Correlação Cruzada")
        ax22.legend()
        self._save_fig(fig22, "22_dinamica_correlacao_cruzada_lead_lag")

        # Plot 23: Espectro de Welch
        freqs_cpd, psd_eff = signal.welch(signal.detrend(s_eff_1h), fs=24.0, nperseg=512)
        _, psd_pwr = signal.welch(signal.detrend(s_pwr_1h), fs=24.0, nperseg=512)
        fig23, ax23 = plt.subplots(figsize=(8.5, 4.5))
        ax23.semilogy(freqs_cpd[1:], psd_eff[1:] / psd_eff[1:].max(), color="#008080", linewidth=1.8, label="PSD Eficiência")
        ax23.semilogy(freqs_cpd[1:], psd_pwr[1:] / psd_pwr[1:].max(), color="#9467bd", linewidth=1.4, alpha=0.8, label="PSD Potência")
        ax23.axvline(1.0, color="crimson", linestyle="--", label="1 ciclo/dia (24h)")
        ax23.axvline(2.0, color="darkorange", linestyle=":", label="2 ciclos/dia (12h)")
        ax23.set_xlim(0, 6)
        ax23.set_title("Densidade Espectral de Potência (Método de Welch)")
        ax23.set_xlabel("Frequência (ciclos por dia)")
        ax23.set_ylabel("Potência Espectral Relativa (log)")
        ax23.legend()
        self._save_fig(fig23, "23_dinamica_espectro_welch_periodicidades")

    def plot_multivariate_and_maps(self, df_norm_only: pd.DataFrame) -> None:
        """Gera matriz de correlação, dendrograma, mapas de performance e PCA."""
        self.logger.info("5/8. Executando análises multivariadas, mapas físicos e PCA...")
        df_norm_valid = df_norm_only.dropna()
        sample_map = df_norm_valid.sample(n=min(30_000, len(df_norm_valid)), random_state=RANDOM_SEED)

        # Plot 24A: Matriz de Correlação Linear de Pearson
        corr_pearson = df_norm_valid.corr(method="pearson")
        fig24, ax24 = plt.subplots(figsize=(10, 8.5))
        mask_tri = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)
        sns.heatmap(corr_pearson, mask=mask_tri, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    annot_kws={"size": 8.0}, ax=ax24, cbar_kws={"shrink": 0.8, "label": "Coeficiente de Pearson (r)"})
        ax24.set_title("Matriz de Correlação Linear de Pearson — 12 Sensores CBM (Operação Normal)", fontsize=11, fontweight="bold")
        self._save_fig(fig24, "24_matriz_correlacao_pearson")

        # Plot 24B: Matriz de Correlação Monotônica de Spearman
        corr_spearman = df_norm_valid.sample(n=min(50_000, len(df_norm_valid)), random_state=RANDOM_SEED).corr(method="spearman")
        fig24_sp, ax24_sp = plt.subplots(figsize=(10, 8.5))
        sns.heatmap(corr_spearman, mask=mask_tri, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                    annot_kws={"size": 8.0}, ax=ax24_sp, cbar_kws={"shrink": 0.8, "label": "Coeficiente de Spearman (rho)"})
        ax24_sp.set_title("Matriz de Correlação Monotônica de Spearman — 12 Sensores CBM", fontsize=11, fontweight="bold")
        self._save_fig(fig24_sp, "24_matriz_correlacao_spearman")

        # Plot 25: Dendrograma
        dist_mat = np.clip((1.0 - np.abs(corr_pearson.values)), 0.0, None)
        np.fill_diagonal(dist_mat, 0.0)
        linkage_z = hierarchy.linkage(squareform(dist_mat), method="average")
        fig25, ax25 = plt.subplots(figsize=(9, 6.0))
        hierarchy.dendrogram(linkage_z, labels=[LABELS_PT[c] for c in SENSOR_COLS], orientation="right", ax=ax25, color_threshold=0.25)
        ax25.set_title(r"Agrupamento Hierárquico dos Sensores ($1 - |r|$)")
        ax25.set_xlabel("Distância de Dissimilaridade")
        self._save_fig(fig25, "25_agrupamento_hierarquico_dendrograma")

        # Plots 26 a 28: Mapas Termodinâmicos
        fig26, ax26 = plt.subplots(figsize=(8.5, 5.0))
        sc0 = ax26.scatter(sample_map["act_press_ratio"], sample_map["act_poly_head"], c=sample_map["act_poly_eff"], cmap="viridis", s=6, alpha=0.45, rasterized=True)
        exp_s = sample_map.sort_values("exp_press_ratio")
        ax26.plot(exp_s["exp_press_ratio"], exp_s["exp_poly_head"], color="crimson", linewidth=2.0, label="Curva Projeto")
        plt.colorbar(sc0, ax=ax26, label="Eficiência Real (%)")
        ax26.set_title(r"Mapa $r_p \times H_p$ (Real vs. Projeto)")
        ax26.set_xlabel("Razão de Compressão (-)")
        ax26.set_ylabel("Head Politrópico (kJ/kg)")
        ax26.legend(loc="lower right")
        self._save_fig(fig26, "26_mapa_termodinamico_rp_vs_hp")

        fig27, ax27 = plt.subplots(figsize=(8.0, 5.0))
        sc1 = ax27.scatter(sample_map["exp_shaft_power"], sample_map["act_shaft_power"], c=sample_map["shaft_pwr_dev"], cmap="plasma", s=6, alpha=0.45, rasterized=True)
        ax27.plot([7000, 11000], [7000, 11000], "k--", linewidth=1.3, label="Paridade 1:1")
        plt.colorbar(sc1, ax=ax27, label="Desvio Potência (%)")
        ax27.set_title("Potência Real vs. Esperada — Consumo Excedente (~+9.8%)")
        ax27.set_xlabel("Potência Esperada (kW)")
        ax27.set_ylabel("Potência Real (kW)")
        ax27.legend(loc="upper left")
        self._save_fig(fig27, "27_mapa_termodinamico_potencia_paridade")

        fig28, ax28 = plt.subplots(figsize=(8.0, 5.0))
        hb = ax28.hexbin(sample_map["poly_head_dev"], sample_map["poly_eff_dev"], gridsize=45, cmap="YlGnBu", mincnt=1)
        plt.colorbar(hb, ax=ax28, label="Densidade de Pontos")
        ax28.set_title("Acoplamento entre Desvio de Head e Desvio de Eficiência")
        ax28.set_xlabel("Desvio de Head Politrópico (%)")
        ax28.set_ylabel("Desvio de Eficiência Reconstruído (%)")
        self._save_fig(fig28, "28_mapa_termodinamico_head_vs_eff")

        # Plots 29 a 31: PCA
        scaler_pca = StandardScaler()
        X_norm_scaled = scaler_pca.fit_transform(df_norm_valid)
        pca_full = PCA(random_state=RANDOM_SEED)
        X_pca_scores = pca_full.fit_transform(X_norm_scaled)
        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        fig29, ax29 = plt.subplots(figsize=(8.0, 4.5))
        pcs_idx = np.arange(1, len(exp_var) + 1)
        ax29.bar(pcs_idx, exp_var * 100, alpha=0.75, color="#2b8cbe", label="Individual (%)")
        ax29.plot(pcs_idx, cum_var * 100, marker="o", color="crimson", linewidth=2, label="Acumulada (%)")
        ax29.axhline(95, color="gray", linestyle="--", label="Limiar 95%")
        ax29.set_title("Variância Explicada por Componente Principal (PCA)")
        ax29.set_xlabel("Componente Principal")
        ax29.set_ylabel("% da Variância")
        ax29.legend()
        self._save_fig(fig29, "29_pca_scree_plot_variancia")

        fig30, ax30 = plt.subplots(figsize=(6.5, 6.0))
        loadings = pd.DataFrame(pca_full.components_[:2].T, index=[c.replace("_", " ") for c in SENSOR_COLS], columns=["PC1", "PC2"])
        sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax30, cbar=False)
        ax30.set_title(f"Loadings de PC1 ({exp_var[0]*100:.1f}%) e PC2 ({exp_var[1]*100:.1f}%)")
        self._save_fig(fig30, "30_pca_loadings_heatmap")

        fig31, ax31 = plt.subplots(figsize=(8.5, 5.0))
        idx_sub = np.linspace(0, len(X_pca_scores) - 1, 20_000, dtype=int)
        time_num = mdates.date2num(df_norm_valid.index[idx_sub])
        sc_pca = ax31.scatter(X_pca_scores[idx_sub, 0], X_pca_scores[idx_sub, 1], c=time_num, cmap="Spectral", s=5, alpha=0.45, rasterized=True)
        cbar = plt.colorbar(sc_pca, ax=ax31)
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax31.set_title("Trajetória Temporal no Espaço Latente (PC1 × PC2)")
        ax31.set_xlabel("PC1 (Desvio Global / Carga Esperada)")
        ax31.set_ylabel("PC2 (Nível de Potência e Pressão Real)")
        self._save_fig(fig31, "31_pca_projecao_latente_tempo")

        # Plot 32: Regimes Operacionais por K-Means
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_SEED)
        raw_labels = kmeans.fit_predict(X_norm_scaled)
        regime_order = pd.Series(df_norm_valid["act_poly_eff"].values).groupby(raw_labels).mean().sort_values(ascending=False).index
        label_remap = {old: f"Regime {new+1}" for new, old in enumerate(regime_order)}
        regimes_series = pd.Series([label_remap[r] for r in raw_labels], index=df_norm_valid.index)
        regime_1h = regimes_series.resample("1h").agg(lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan).dropna()
        eff_1h_aligned = df_norm_valid["act_poly_eff"].resample("1h").mean().loc[regime_1h.index]

        fig32, ax32 = plt.subplots(figsize=(12, 4.5))
        reg_colors = {"Regime 1": "#1b9e77", "Regime 2": "#377eb8", "Regime 3": "#e41a1c"}
        for reg_name, cor in reg_colors.items():
            m = regime_1h == reg_name
            ax32.scatter(eff_1h_aligned.index[m], eff_1h_aligned[m], s=8, color=cor, alpha=0.75, label=reg_name)
        ax32.set_title("Evolução dos Regimes Operacionais (K-Means Multivariado) sobre a Eficiência (1h)")
        ax32.set_ylabel("Eficiência Politrópica Real (%)")
        ax32.set_xlabel("Data")
        ax32.legend(markerscale=2.5, loc="upper right")
        self._save_fig(fig32, "32_regimes_operacionais_kmeans_tempo")

    def plot_anomalies_and_cusum(self, df_5min: pd.DataFrame, df_1d: pd.DataFrame, df_1h: pd.DataFrame, df_norm_only: pd.DataFrame) -> None:
        """Gera detecção de anomalias por consenso, PCA-SPE, carta CUSUM e volatilidade."""
        self.logger.info("6/8. Detectando anomalias e monitorando deriva estrutural (CUSUM)...")
        df_anom = df_5min.dropna().copy()
        X_anom_scaled = RobustScaler().fit_transform(df_anom)

        med_roll = df_anom["act_poly_eff"].rolling(288, center=True, min_periods=36).median()
        mad_roll = (df_anom["act_poly_eff"] - med_roll).abs().rolling(288, center=True, min_periods=36).median() * 1.4826
        z_local = ((df_anom["act_poly_eff"] - med_roll) / mad_roll.replace(0, np.nan)).fillna(0.0)
        flag_univar = z_local.abs() > 4.0

        iso = IsolationForest(n_estimators=150, max_samples=1024, contamination=0.01, random_state=RANDOM_SEED, n_jobs=-1)
        flag_iso = pd.Series(iso.fit_predict(X_anom_scaled) == -1, index=df_anom.index)

        scaler_pca = StandardScaler().fit(df_norm_only.dropna())
        pca_fit = PCA(n_components=3, random_state=RANDOM_SEED).fit(scaler_pca.transform(df_norm_only.dropna()))
        X_std = scaler_pca.transform(df_anom)
        X_rec = pca_fit.transform(X_std) @ pca_fit.components_
        spe_5m = pd.Series(np.sum((X_std - X_rec) ** 2, axis=1), index=df_anom.index)
        spe_thr = spe_5m.quantile(0.99)
        flag_spe = spe_5m > spe_thr

        flag_consensus = (flag_univar.astype(int) + flag_iso.astype(int) + flag_spe.astype(int)) >= 2

        # Plot 33: Anomalias de Consenso
        fig33, ax33 = plt.subplots(figsize=(12, 4.5))
        ax33.plot(df_anom.index, df_anom["act_poly_eff"], color="#008080", linewidth=0.7, alpha=0.8, label="Eficiência Real (5 min)")
        ax33.scatter(df_anom.index[flag_consensus], df_anom.loc[flag_consensus, "act_poly_eff"], color="crimson", s=18, zorder=5, label=f"Anomalias Consenso (n={flag_consensus.sum()})")
        ax33.set_title("Anomalias Multivariadas por Consenso Destacadas sobre a Eficiência Politrópica")
        ax33.set_ylabel("Eficiência (%)")
        ax33.legend(loc="upper right")
        self._save_fig(fig33, "33_anomalias_consenso_eficiencia")

        # Plot 34: PCA-SPE
        fig34, ax34 = plt.subplots(figsize=(12, 4.2))
        ax34.plot(spe_5m.index, spe_5m, color="#7570b3", linewidth=0.8, label="Estatística SPE / Q")
        ax34.axhline(spe_thr, color="crimson", linestyle="--", linewidth=1.2, label=f"Limiar P99 ({spe_thr:.2f})")
        ax34.set_yscale("log")
        ax34.set_title("Monitoramento de Quebra de Coerência Física — PCA-SPE (Escala Log)")
        ax34.set_ylabel("SPE / Q (log)")
        ax34.legend(loc="upper right")
        self._save_fig(fig34, "34_anomalias_pca_spe_q_statistic")

        # Plots 35 e 36: CUSUM e Volatilidade
        eff_daily = df_1d["act_poly_eff"].dropna()
        mu_ref, sigma_ref = eff_daily.iloc[:14].mean(), eff_daily.std()
        k_slack = 0.25 * sigma_ref
        c_neg, c_pos = np.zeros(len(eff_daily)), np.zeros(len(eff_daily))
        for t in range(1, len(eff_daily)):
            c_pos[t] = max(0.0, c_pos[t-1] + (eff_daily.iloc[t] - mu_ref) - k_slack)
            c_neg[t] = max(0.0, c_neg[t-1] - (eff_daily.iloc[t] - mu_ref) - k_slack)

        fig35, ax35 = plt.subplots(figsize=(11, 4.2))
        ax35.plot(eff_daily.index, c_neg, color="crimson", linewidth=2.2, label=r"CUSUM Inferior ($C^-$: Perda de Eficiência)")
        ax35.plot(eff_daily.index, c_pos, color="seagreen", linewidth=1.6, label=r"CUSUM Superior ($C^+$)")
        ax35.axvline(pd.Timestamp("2019-11-21"), color="navy", linestyle="--", label="Inflexão / Recuperação (~21/Nov)")
        ax35.set_title("Carta CUSUM sobre a Eficiência Diária (Referência = Primeiras 2 Semanas Set/2019)")
        ax35.set_ylabel("Soma Acumulada")
        ax35.legend(loc="upper left")
        self._save_fig(fig35, "35_cusum_deriva_estrutural_eficiencia")

        vol_24h = df_1h["act_shaft_power"].rolling(24, min_periods=6).std()
        fig36, ax36 = plt.subplots(figsize=(11, 4.0))
        ax36.plot(vol_24h.index, vol_24h, color="#d95f02", linewidth=1.1)
        ax36.set_title("Volatilidade Operacional — Desvio Padrão Móvel de 24h da Potência de Eixo (kW)")
        ax36.set_ylabel(r"$\sigma_{24h}$ Potência (kW)")
        self._save_fig(fig36, "36_volatilidade_movel_potencia_24h")

    def run(self) -> None:
        """Executa a rotina completa de ponta a ponta."""
        self.logger.info("Iniciando pipeline de Análise Exploratória (EDA)...")
        df, df_30s, state = self.load_and_clean_data()
        df_clean_30s, df_5min, df_1h, df_1d = self.plot_macro_and_micro(df_30s, state)
        df_norm_only = df_30s.where(state == "Operação Normal")
        self.plot_distributions_and_seasonality(df_norm_only, df_1h)
        self.plot_temporal_dynamics(df_clean_30s, df_5min, df_1h)
        self.plot_multivariate_and_maps(df_norm_only)
        self.plot_anomalies_and_cusum(df_5min, df_1d, df_1h, df_norm_only)
        self.logger.info("Pipeline de EDA concluído com sucesso! Gráficos salvos em: %s", self.output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Script de Geração de Plots e Análise Exploratória de Dados (EDA) do Compressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(repo_root / "data" / "compressor.csv"),
        help="Caminho para o arquivo compressor.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "results" / "eda"),
        help="Diretório onde os arquivos PDF individuais serão salvos",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Formato dos arquivos gráficos gerados",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = CompressorEDAPipeline(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        plot_format=args.format,
    )
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

