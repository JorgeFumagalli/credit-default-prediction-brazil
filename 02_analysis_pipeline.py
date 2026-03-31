#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
02_analysis_pipeline.py
TCC — Previsão da Inadimplência de Cartões de Crédito no Brasil

SCRIPT 2: MODELAGEM, DIAGNÓSTICOS DOS MODELOS E TESTE DE CHOW
==============================================================================

Este script consolida as etapas analíticas finais:
    1. Modelagem preditiva nos cenários FULL e EXCL
    2. Diagnósticos dos modelos lineares e do XGBoost
    3. Teste de Breusch-Pagan
    4. Teste de Chow para quebra estrutural
    5. Geração de gráficos e comparações FULL vs EXCL

Modelos:
    - Regressão Linear (OLS + Stepwise + Box-Cox)
    - ARIMA / SARIMAX
    - Random Forest
    - XGBoost
    - MLP
    - LSTM
============================================================================== 
"""

# %% ==================== IMPORTAÇÕES ====================
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf

from scipy.special import inv_boxcox
from scipy.stats import boxcox, f
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# %% ==================== REPRODUTIBILIDADE ====================
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# %% ==================== CONFIGURAÇÕES ====================
PREP_DIR = Path("./prepared")
RES_DIR = Path("./results")
DIAG_DIR = RES_DIR / "diagnostics"

for directory in [RES_DIR, DIAG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "inadimpl_cartao_total"
DATE_COL = "data"

DROP_COLS_RULE = ["endividamento_familias", "ibcbr_sem_ajuste"]

STEPWISE_P_IN = 0.05
STEPWISE_P_OUT = 0.10
BOXCOX_EPS = 1e-6

BREAK_DATES = [
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
]


# %% ==================== MÉTRICAS ====================
def mape(y_true, y_pred, eps=1e-9):
    """Mean Absolute Percentage Error em %."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def direction_acc(y_true, y_pred):
    """Directional Accuracy: percentual de acertos na direção da variação."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    return ((true_diff > 0) == (pred_diff > 0)).mean() * 100


def r2_variance_score(y_true, y_pred):
    """
    R² da variância:
        ESS / (ESS + RSS)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    y_bar = np.mean(y_true)
    ess = np.sum((y_pred - y_bar) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    denom = ess + rss

    if np.isclose(denom, 0):
        return np.nan

    return ess / denom


def adjusted_r2_score(y_true, y_pred, p):
    """
    R² ajustado:
        1 - (1 - R²) * (n - 1) / (n - p - 1)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    if n <= p + 1:
        return np.nan

    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def evaluate(y_true, y_pred, model_name, p_used):
    """Consolida as métricas de avaliação do modelo."""
    return dict(
        Model=model_name,
        MSE=mean_squared_error(y_true, y_pred),
        R2_adjust=adjusted_r2_score(y_true, y_pred, p_used),
        R2_variance=r2_variance_score(y_true, y_pred),
        MAPE=mape(y_true, y_pred),
        DA=direction_acc(y_true, y_pred),
        N_Features=p_used,
    )


# %% ==================== MODELOS KERAS ====================
def make_mlp_model(input_dim: int) -> Model:
    """Cria modelo MLP."""
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


def make_lstm_model(input_shape) -> Model:
    """Cria modelo LSTM."""
    inp = Input(shape=input_shape)
    x = LSTM(64, activation="tanh")(inp)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


# %% ==================== VISUALIZAÇÕES ====================
def format_abnt_axes(ax, xlabel: str = "", ylabel: str = "", show_legend: bool = True):
    """Formata os eixos com estilo consistente com o TCC."""
    if xlabel:
        ax.set_xlabel(xlabel, fontfamily="Arial", fontsize=11, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontfamily="Arial", fontsize=11, color="black")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontsize(9)
        label.set_color("black")

    ax.grid(False)
    ax.set_facecolor("white")
    if ax.figure is not None:
        ax.figure.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color("black")

    if show_legend and len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(frameon=False, prop={"family": "Arial", "size": 9})


def plot_real_pred(dates, y_true, y_pred, outpath):
    """Gera gráfico Real vs Predito."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, y_true, label="Real", linewidth=2)
    ax.plot(dates, y_pred, label="Predito", linestyle="--", linewidth=2)

    format_abnt_axes(
        ax,
        xlabel="Data",
        ylabel="Taxa de inadimplência (%)",
        show_legend=True,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bars(dfres: pd.DataFrame, tag: str, outdir: Path):
    """Gera gráficos de barras para as métricas do cenário."""
    metrics = ["MSE", "R2_adjust", "R2_variance", "MAPE", "DA"]

    ylabel_map = {
        "MSE": "MSE",
        "R2_adjust": "R² ajustado",
        "R2_variance": "R² da variância",
        "MAPE": "MAPE (%)",
        "DA": "DA (%)",
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(dfres["Model"], dfres[metric])

        format_abnt_axes(
            ax,
            xlabel="Modelo",
            ylabel=ylabel_map.get(metric, metric),
            show_legend=False,
        )

        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(outdir / f"{tag}_metric_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_full_vs_excl_series(df_full: pd.DataFrame, df_excl: pd.DataFrame, outdir: Path):
    """Compara visualmente a série target nos cenários FULL e EXCL."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_full[DATE_COL], df_full[TARGET], label="FULL", linewidth=2)
    ax.plot(df_excl[DATE_COL], df_excl[TARGET], label="EXCL (sem 2019–2021)", linewidth=2)

    format_abnt_axes(
        ax,
        xlabel="Data",
        ylabel="Taxa de inadimplência (%)",
        show_legend=True,
    )

    plt.tight_layout()
    plt.savefig(outdir / "FULL_vs_EXCL_inadimplencia.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compare_metric_full_excl(df_full_res: pd.DataFrame, df_excl_res: pd.DataFrame, metric: str, outdir: Path):
    """Compara uma métrica entre FULL e EXCL por modelo."""
    mf = df_full_res[["Model", metric]].rename(columns={metric: f"{metric}_FULL"})
    me = df_excl_res[["Model", metric]].rename(columns={metric: f"{metric}_EXCL"})
    merged = mf.merge(me, on="Model", how="inner")

    x = np.arange(len(merged["Model"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, merged[f"{metric}_FULL"], width, label="FULL")
    ax.bar(x + width / 2, merged[f"{metric}_EXCL"], width, label="EXCL")

    ax.set_xticks(x)
    ax.set_xticklabels(merged["Model"], rotation=20)

    ylabel_map = {
        "MSE": "MSE",
        "R2_adjust": "R² ajustado",
        "R2_variance": "R² da variância",
        "MAPE": "MAPE (%)",
        "DA": "DA (%)",
    }

    format_abnt_axes(
        ax,
        xlabel="Modelo",
        ylabel=ylabel_map.get(metric, metric),
        show_legend=True,
    )

    plt.tight_layout()
    plt.savefig(outdir / f"COMPARE_FULL_EXCL_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel_full_excl(df_full_res: pd.DataFrame, df_excl_res: pd.DataFrame, outdir: Path):
    """
    Gera painel consolidado FULL vs EXCL na ordem:
        1) R² da variância
        2) MSE
        3) MAPE
        4) DA
    """
    metrics = ["R2_variance", "MSE", "MAPE", "DA"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    ylabel_map = {
        "MSE": "MSE",
        "R2_variance": "R² da variância",
        "MAPE": "MAPE (%)",
        "DA": "DA (%)",
    }

    for i, metric in enumerate(metrics):
        ax = axes[i]

        mf = df_full_res[["Model", metric]].rename(columns={metric: f"{metric}_FULL"})
        me = df_excl_res[["Model", metric]].rename(columns={metric: f"{metric}_EXCL"})
        merged = mf.merge(me, on="Model", how="inner")

        x = np.arange(len(merged["Model"]))
        width = 0.35

        ax.bar(x - width / 2, merged[f"{metric}_FULL"], width, label="FULL")
        ax.bar(x + width / 2, merged[f"{metric}_EXCL"], width, label="EXCL")

        ax.set_xticks(x)
        ax.set_xticklabels(merged["Model"], rotation=20)

        format_abnt_axes(
            ax,
            xlabel="Modelo",
            ylabel=ylabel_map.get(metric, metric),
            show_legend=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        prop={"family": "Arial", "size": 10},
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(outdir / "PANEL_FULL_EXCL_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_model(dates, y_true, y_pred, tag: str, model_name: str, outdir: Path):
    """Salva o gráfico do melhor modelo por cenário."""
    outpath = outdir / f"{tag}_best_model_{model_name}_real_vs_pred.png"
    plot_real_pred(dates, y_true, y_pred, outpath)


def plot_chow_breaks(df: pd.DataFrame, outdir: Path):
    """Plota a série target com as datas candidatas de quebra estrutural."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df[DATE_COL], df[TARGET], linewidth=2, label="Inadimplência observada")

    for break_date in BREAK_DATES:
        ax.axvline(pd.to_datetime(break_date), linestyle="--", linewidth=1)

    format_abnt_axes(
        ax,
        xlabel="Data",
        ylabel="Taxa de inadimplência (%)",
        show_legend=True,
    )

    plt.tight_layout()
    plt.savefig(outdir / "chow_breakpoints_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# %% ==================== UTILIDADES DE FEATURES ====================
def load_prepared_datasets():
    """Carrega os datasets FULL e EXCL preparados no script 1."""
    full_path = PREP_DIR / "prepared_FULL.parquet"
    excl_path = PREP_DIR / "prepared_EXCL.parquet"

    if not full_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {full_path}")

    df_full = pd.read_parquet(full_path)
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])
    df_full = df_full.sort_values(DATE_COL).reset_index(drop=True)

    if excl_path.exists():
        df_excl = pd.read_parquet(excl_path)
        df_excl[DATE_COL] = pd.to_datetime(df_excl[DATE_COL])
        df_excl = df_excl.sort_values(DATE_COL).reset_index(drop=True)
    else:
        mask_excl = (df_full[DATE_COL] < "2019-01-01") | (df_full[DATE_COL] > "2021-12-01")
        df_excl = df_full.loc[mask_excl].reset_index(drop=True)

    return df_full, df_excl


def get_feature_columns(df: pd.DataFrame):
    """Retorna colunas de features."""
    return [c for c in df.columns if c not in [DATE_COL, TARGET]]


def remove_rule_cols(cols):
    """Remove colunas excluídas pela regra metodológica."""
    return [c for c in cols if c not in DROP_COLS_RULE]


# %% ==================== STEPWISE ====================
def stepwise_selection(X: pd.DataFrame, y: pd.Series, initial_features=None,
                       threshold_in=0.05, threshold_out=0.10, verbose=True):
    """
    Stepwise bidirecional baseado em p-valores do OLS.
    """
    if initial_features is None:
        included = []
    else:
        included = list(initial_features)

    all_features = list(X.columns)

    while True:
        changed = False

        excluded = list(set(all_features) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)

        for new_col in excluded:
            try:
                model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
                new_pvals.loc[new_col] = model.pvalues[new_col]
            except Exception:
                new_pvals.loc[new_col] = np.nan

        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if pd.notna(best_pval) and best_pval < threshold_in:
                best_feature = new_pvals.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"  Stepwise + adicionada: {best_feature} (p={best_pval:.6f})")

        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()

            if pd.notna(worst_pval) and worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"  Stepwise - removida: {worst_feature} (p={worst_pval:.6f})")

        if not changed:
            break

    return included


# %% ==================== LINEAR: BOX-COX + BP ====================
def fit_linear_with_boxcox(Xtr_df: pd.DataFrame, ytr: np.ndarray, selected_features):
    """Ajusta o modelo linear no espaço Box-Cox e roda Breusch-Pagan."""
    if len(selected_features) == 0:
        raise ValueError("Nenhuma variável selecionada para o modelo linear.")

    Xtr_sel = Xtr_df[selected_features].copy()

    y_shift = 0.0
    ytr_min = float(np.min(ytr))
    if ytr_min <= 0:
        y_shift = abs(ytr_min) + BOXCOX_EPS

    ytr_pos = ytr + y_shift + BOXCOX_EPS
    ytr_boxcox, lambda_bc = boxcox(ytr_pos)

    Xtr_const = sm.add_constant(Xtr_sel, has_constant="add")
    ols_model = sm.OLS(ytr_boxcox, Xtr_const).fit()

    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(ols_model.resid, Xtr_const)

    bp_result = {
        "bp_lm_stat": bp_lm,
        "bp_lm_pvalue": bp_lm_pvalue,
        "bp_f_stat": bp_fvalue,
        "bp_f_pvalue": bp_f_pvalue,
    }

    return ols_model, lambda_bc, y_shift, bp_result


def predict_linear_with_boxcox(model, X_df: pd.DataFrame, selected_features, lambda_bc, y_shift):
    """Gera predições do modelo linear e reverte Box-Cox."""
    Xte_sel = X_df[selected_features].copy()
    Xte_const = sm.add_constant(Xte_sel, has_constant="add")

    pred_bc = model.predict(Xte_const)
    pred = inv_boxcox(pred_bc, lambda_bc) - y_shift - BOXCOX_EPS
    return np.array(pred, dtype=float)


# %% ==================== ARIMA / SARIMAX ====================
def select_best_arima_order(y_train, exog_train=None, p_values=(0, 1, 2, 3),
                            d_values=(0, 1, 2), q_values=(0, 1, 2, 3)):
    """Seleciona a melhor ordem ARIMA via AIC."""
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = SARIMAX(
                        y_train,
                        exog=exog_train,
                        order=(p, d, q),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except Exception:
                    continue

    return best_order, best_model, best_aic


# %% ==================== TESTE DE CHOW ====================
def prepare_features_for_chow(df: pd.DataFrame):
    """Prepara X e y para o teste de Chow."""
    cols_to_drop = [DATE_COL, TARGET] + [c for c in DROP_COLS_RULE if c in df.columns]
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y


def fit_ols_and_get_rss(X: pd.DataFrame, y: pd.Series):
    """Ajusta OLS com intercepto e retorna RSS, n e k."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()
    rss = float(np.sum(model.resid ** 2))
    n = int(len(y))
    k = int(X_const.shape[1])
    return model, rss, n, k


def chow_test(df: pd.DataFrame, break_date: str) -> dict:
    """Executa o teste de Chow para uma data de quebra específica."""
    break_date = pd.to_datetime(break_date)

    df_pre = df[df[DATE_COL] < break_date].copy()
    df_post = df[df[DATE_COL] >= break_date].copy()

    if len(df_pre) < 10 or len(df_post) < 10:
        return {
            "break_date": break_date.date().isoformat(),
            "n_pre": len(df_pre),
            "n_post": len(df_post),
            "k_params": np.nan,
            "rss_pooled": np.nan,
            "rss_pre": np.nan,
            "rss_post": np.nan,
            "chow_f": np.nan,
            "p_value": np.nan,
            "significant_5pct": np.nan,
            "note": "Amostra insuficiente em uma das subamostras.",
        }

    X_all, y_all = prepare_features_for_chow(df)
    _, rss_pooled, _, k = fit_ols_and_get_rss(X_all, y_all)

    X_pre, y_pre = prepare_features_for_chow(df_pre)
    _, rss_pre, n_pre, _ = fit_ols_and_get_rss(X_pre, y_pre)

    X_post, y_post = prepare_features_for_chow(df_post)
    _, rss_post, n_post, _ = fit_ols_and_get_rss(X_post, y_post)

    denominator_df = (n_pre + n_post - 2 * k)
    if denominator_df <= 0:
        return {
            "break_date": break_date.date().isoformat(),
            "n_pre": n_pre,
            "n_post": n_post,
            "k_params": k,
            "rss_pooled": rss_pooled,
            "rss_pre": rss_pre,
            "rss_post": rss_post,
            "chow_f": np.nan,
            "p_value": np.nan,
            "significant_5pct": np.nan,
            "note": "Graus de liberdade insuficientes para o teste.",
        }

    numerator = (rss_pooled - (rss_pre + rss_post)) / k
    denominator = (rss_pre + rss_post) / denominator_df
    chow_f = numerator / denominator
    p_value = 1 - f.cdf(chow_f, k, denominator_df)

    return {
        "break_date": break_date.date().isoformat(),
        "n_pre": n_pre,
        "n_post": n_post,
        "k_params": k,
        "rss_pooled": rss_pooled,
        "rss_pre": rss_pre,
        "rss_post": rss_post,
        "chow_f": chow_f,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
        "note": "",
    }


def run_chow_tests(df_full: pd.DataFrame):
    """Roda o teste principal e os testes complementares de Chow."""
    main_break = "2020-01-01"
    result_main = chow_test(df_full, main_break)
    df_main = pd.DataFrame([result_main])
    df_main.to_csv(RES_DIR / "chow_test_single_break.csv", index=False)

    results_multi = [chow_test(df_full, d) for d in BREAK_DATES]
    df_multi = pd.DataFrame(results_multi)
    df_multi.to_csv(RES_DIR / "chow_test_multiple_breaks_2019_2021.csv", index=False)

    plot_chow_breaks(df_full, RES_DIR)

    return df_main, df_multi


# %% ==================== ANÁLISE POR CENÁRIO ====================
def run_period_analysis(df: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, dict]:
    """Executa a modelagem completa para um cenário (FULL ou EXCL)."""
    print(f"\n=== Análise {tag} ===")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    feats_all = get_feature_columns(df)

    X_all = df[feats_all].copy()
    y = df[TARGET].values
    dates = df[DATE_COL].values

    cut = int(len(df) * 0.8)
    Xtr_all_df = X_all.iloc[:cut].copy()
    Xte_all_df = X_all.iloc[cut:].copy()
    ytr = y[:cut]
    yte = y[cut:]
    dte = dates[cut:]

    scaler_all = StandardScaler()
    Xtr_all_s = scaler_all.fit_transform(Xtr_all_df.values)
    Xte_all_s = scaler_all.transform(Xte_all_df.values)

    Xtr_all_s_df = pd.DataFrame(Xtr_all_s, columns=feats_all, index=Xtr_all_df.index)
    Xte_all_s_df = pd.DataFrame(Xte_all_s, columns=feats_all, index=Xte_all_df.index)

    feats_filtered = remove_rule_cols(feats_all)
    Xtr_filtered_df = Xtr_all_df[feats_filtered].copy()
    Xte_filtered_df = Xte_all_df[feats_filtered].copy()

    scaler_filtered = StandardScaler()
    Xtr_filtered_s = scaler_filtered.fit_transform(Xtr_filtered_df.values)
    Xte_filtered_s = scaler_filtered.transform(Xte_filtered_df.values)

    Xtr_filtered_s_df = pd.DataFrame(Xtr_filtered_s, columns=feats_filtered, index=Xtr_filtered_df.index)
    Xte_filtered_s_df = pd.DataFrame(Xte_filtered_s, columns=feats_filtered, index=Xte_filtered_df.index)

    results = []
    preds = {}
    diag_rows = []
    bp_rows = []

    # 1) Stepwise para Linear
    print(f"\n{tag} - Executando Stepwise...")
    selected_stepwise_features = stepwise_selection(
        X=Xtr_all_s_df,
        y=pd.Series(ytr, index=Xtr_all_s_df.index),
        threshold_in=STEPWISE_P_IN,
        threshold_out=STEPWISE_P_OUT,
        verbose=True,
    )

    if len(selected_stepwise_features) == 0:
        raise RuntimeError(f"{tag} - O stepwise não selecionou variáveis.")

    selected_linear_features = remove_rule_cols(selected_stepwise_features)
    if len(selected_linear_features) == 0:
        raise RuntimeError(f"{tag} - Após a regra metodológica, não restaram preditores para o modelo linear.")

    print(f"{tag} - Variáveis finais do Linear: {selected_linear_features}")

    # 2) Linear (OLS + Box-Cox + BP)
    linear_model, lambda_bc, y_shift, bp_result = fit_linear_with_boxcox(
        Xtr_df=Xtr_all_s_df,
        ytr=ytr,
        selected_features=selected_linear_features,
    )

    preds["Linear"] = predict_linear_with_boxcox(
        model=linear_model,
        X_df=Xte_all_s_df,
        selected_features=selected_linear_features,
        lambda_bc=lambda_bc,
        y_shift=y_shift,
    )

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["Linear"],
            model_name="Linear",
            p_used=len(selected_linear_features),
        )
    )

    # Diagnóstico linear
    coef_values = linear_model.params.drop(labels="const", errors="ignore")
    df_coef = pd.DataFrame({
        "Variavel": coef_values.index,
        "Coef_padronizado": coef_values.values,
    })
    df_coef["Abs_Coef"] = df_coef["Coef_padronizado"].abs()
    df_coef["Cenario"] = tag
    df_coef = df_coef.sort_values("Abs_Coef", ascending=False).reset_index(drop=True)
    df_coef.to_csv(DIAG_DIR / f"linear_coeffs_{tag}.csv", index=False, float_format="%.6f")

    diag_rows.append({
        "Scenario": tag,
        "Model": "Linear",
        "Stepwise_Selected_Before_Rule": " | ".join(selected_stepwise_features),
        "Final_Selected_After_Rule": " | ".join(selected_linear_features),
        "N_Selected_Final": len(selected_linear_features),
        "BoxCox_Lambda": lambda_bc,
        "Y_Shift": y_shift,
        "Train_R2_OLS_BoxCox": linear_model.rsquared,
        "Train_R2Adj_OLS_BoxCox": linear_model.rsquared_adj,
        "AIC": linear_model.aic,
        "BIC": linear_model.bic,
    })

    bp_rows.append({
        "Scenario": tag,
        "Model": "Linear",
        "bp_lm_stat": bp_result["bp_lm_stat"],
        "bp_lm_pvalue": bp_result["bp_lm_pvalue"],
        "bp_f_stat": bp_result["bp_f_stat"],
        "bp_f_pvalue": bp_result["bp_f_pvalue"],
    })

    # 3) ARIMA / SARIMAX
    best_order, arima_model, best_aic = select_best_arima_order(
        y_train=ytr,
        exog_train=Xtr_filtered_s_df,
    )

    if arima_model is None:
        raise RuntimeError(f"{tag} - Não foi possível ajustar um ARIMA/SARIMAX válido.")

    preds["ARIMA"] = np.array(arima_model.forecast(steps=len(yte), exog=Xte_filtered_s_df)).flatten()

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["ARIMA"],
            model_name="ARIMA",
            p_used=Xtr_filtered_s_df.shape[1],
        )
    )

    diag_rows.append({
        "Scenario": tag,
        "Model": "ARIMA",
        "Exogenous_Features": " | ".join(feats_filtered),
        "N_Selected_Final": len(feats_filtered),
        "ARIMA_Order": str(best_order),
        "AIC": best_aic,
        "BIC": getattr(arima_model, "bic", np.nan),
    })

    print(f"{tag} - Melhor ARIMA: ordem={best_order} | AIC={best_aic:.4f}")

    # 4) Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xtr_filtered_df.values, ytr)
    preds["RandomForest"] = rf.predict(Xte_filtered_df.values)

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["RandomForest"],
            model_name="RandomForest",
            p_used=Xtr_filtered_df.shape[1],
        )
    )

    # 5) XGBoost
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    xgb.fit(Xtr_filtered_df.values, ytr)
    preds["XGBoost"] = xgb.predict(Xte_filtered_df.values)

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["XGBoost"],
            model_name="XGBoost",
            p_used=Xtr_filtered_df.shape[1],
        )
    )

    df_imp = pd.DataFrame({
        "Variavel": feats_filtered,
        "Importancia": xgb.feature_importances_,
    })
    df_imp["Importancia_pct"] = df_imp["Importancia"] / df_imp["Importancia"].sum() * 100
    df_imp["Cenario"] = tag
    df_imp = df_imp.sort_values("Importancia", ascending=False).reset_index(drop=True)
    df_imp.to_csv(DIAG_DIR / f"xgb_importance_{tag}.csv", index=False, float_format="%.6f")

    # 6) MLP
    mlp = make_mlp_model(Xtr_filtered_s.shape[1])
    es_mlp = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    mlp.fit(
        Xtr_filtered_s,
        ytr,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[es_mlp],
    )

    preds["MLP"] = mlp.predict(Xte_filtered_s, verbose=0).flatten()

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["MLP"],
            model_name="MLP",
            p_used=Xtr_filtered_s.shape[1],
        )
    )

    # 7) LSTM
    Xtr_r = Xtr_filtered_s.reshape((Xtr_filtered_s.shape[0], 1, Xtr_filtered_s.shape[1]))
    Xte_r = Xte_filtered_s.reshape((Xte_filtered_s.shape[0], 1, Xte_filtered_s.shape[1]))

    lstm = make_lstm_model((1, Xtr_filtered_s.shape[1]))
    es_lstm = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    lstm.fit(
        Xtr_r,
        ytr,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[es_lstm],
    )

    preds["LSTM"] = lstm.predict(Xte_r, verbose=0).flatten()

    results.append(
        evaluate(
            y_true=yte,
            y_pred=preds["LSTM"],
            model_name="LSTM",
            p_used=Xtr_filtered_s.shape[1],
        )
    )

    # Consolidação de resultados
    dfres = pd.DataFrame(results).sort_values(by="R2_variance", ascending=False).reset_index(drop=True)
    dfres.to_csv(RES_DIR / f"results_{tag}_final.csv", index=False, float_format="%.4f")

    pd.DataFrame(diag_rows).to_csv(RES_DIR / f"{tag}_linear_diagnostics.csv", index=False)
    pd.DataFrame(bp_rows).to_csv(RES_DIR / f"{tag}_breusch_pagan.csv", index=False)

    print(f"\n{tag} - Resultados finais:")
    print(dfres.to_string(index=False, float_format="%.4f"))

    best_model = dfres.iloc[0]["Model"]
    print(f"\nMelhor modelo ({tag}) por R² da variância: {best_model}")

    # Gráficos por modelo
    for model_name, y_pred in preds.items():
        plot_real_pred(dte, yte, y_pred, RES_DIR / f"{tag}_{model_name}_real_vs_pred.png")

    # Gráfico apenas do campeão
    plot_best_model(dte, yte, preds[best_model], tag, best_model, RES_DIR)

    # Métricas em barras
    plot_metrics_bars(dfres, tag, RES_DIR)

    context = {
        "dates_test": dte,
        "y_test": yte,
        "preds": preds,
        "best_model": best_model,
    }

    return dfres, context


# %% ==================== EXECUÇÃO PRINCIPAL ====================
def main():
    print("\n" + "=" * 80)
    print("TCC - PREVISÃO DA INADIMPLÊNCIA DE CARTÕES DE CRÉDITO NO BRASIL")
    print("PIPELINE 2: MODELAGEM, DIAGNÓSTICOS E TESTE DE CHOW")
    print("=" * 80)

    df_full, df_excl = load_prepared_datasets()

    print(f"Observações FULL: {len(df_full)}")
    print(f"Observações EXCL (sem 2019–2021): {len(df_excl)}")

    dfres_full, _ = run_period_analysis(df_full, "FULL")
    dfres_excl, _ = run_period_analysis(df_excl, "EXCL")

    # Comparações FULL vs EXCL
    plot_full_vs_excl_series(df_full, df_excl, RES_DIR)

    for metric in ["MSE", "R2_adjust", "R2_variance", "MAPE", "DA"]:
        plot_compare_metric_full_excl(dfres_full, dfres_excl, metric, RES_DIR)

    plot_panel_full_excl(dfres_full, dfres_excl, RES_DIR)

    # Consolidação
    df_all = pd.concat([
        dfres_full.assign(Cenario="FULL"),
        dfres_excl.assign(Cenario="EXCL"),
    ], ignore_index=True)
    df_all.to_csv(RES_DIR / "results_FULL_EXCL_consolidated.csv", index=False, float_format="%.4f")

    # Chow
    df_chow_main, df_chow_multi = run_chow_tests(df_full)

    print("\nResultado principal do teste de Chow:")
    print(df_chow_main.to_string(index=False, float_format="%.6f"))

    print("\nResultados para múltiplos pontos de quebra:")
    print(df_chow_multi.to_string(index=False, float_format="%.6f"))

    print("\n" + "=" * 80)
    print("✅ PIPELINE 2 CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print("Arquivos gerados:")
    print(f"   1. {RES_DIR / 'results_FULL_final.csv'}")
    print(f"   2. {RES_DIR / 'results_EXCL_final.csv'}")
    print(f"   3. {RES_DIR / 'results_FULL_EXCL_consolidated.csv'}")
    print(f"   4. {RES_DIR / 'chow_test_single_break.csv'}")
    print(f"   5. {RES_DIR / 'chow_test_multiple_breaks_2019_2021.csv'}")
    print(f"   6. {DIAG_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
