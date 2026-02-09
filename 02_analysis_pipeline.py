#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
02_analysis_pipeline.py
TCC — Previsão da Inadimplência de Cartões de Crédito no Brasil

SCRIPT 2: ANÁLISES, DIAGNÓSTICOS E MODELAGEM OTIMIZADA
==============================================================================

Este script consolida todas as etapas analíticas:
    1. Análise de Colinearidade (Correlação + VIF)
    2. Modelagem Exploratória (5 modelos + diagnósticos)
    3. Modelagem Otimizada Final (features selecionadas)

Autor: Jorge Fumagalli
Data: Janeiro 2026
==============================================================================
"""

#%% ==================== IMPORTS ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from typing import Dict, Tuple, List

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Estatística
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Deep Learning
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Seeds para reprodutibilidade
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

#%% ==================== CONFIGURAÇÕES ====================
# Diretórios
PREP_DIR = Path("./prepared")
COLIN_DIR = Path("./colinearity_results")
PLOT_DIAG_DIR = Path("./plots_diagnostics")
RES_DIAG_DIR = Path("./results_diagnostics")
RES_FINAL_DIR = Path("./results_final")
DIAG_FINAL_DIR = RES_FINAL_DIR / "diagnostics"

# Criar diretórios
for directory in [COLIN_DIR, PLOT_DIAG_DIR, RES_DIAG_DIR, RES_FINAL_DIR, DIAG_FINAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Constantes
TARGET = "inadimpl_cartao_total"
DATE_COL = "data"

# Estilo de gráficos (ABNT)
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (10, 6),
    "axes.edgecolor": "black",
    "axes.linewidth": 1,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})


#%% ==================== FUNÇÕES AUXILIARES ====================

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets FULL e EXCL.
    
    FULL: Série completa (2015-2025)
    EXCL: Exclui período de instabilidade fiscal (2019-2021)
    
    Returns
    -------
    tuple
        (df_full, df_excl)
    """
    path = PREP_DIR / "prepared_FULL.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {path}")
    
    df_full = pd.read_parquet(path)
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])
    df_full = df_full.sort_values(DATE_COL).reset_index(drop=True)
    
    # Cenário EXCL: remove 2019-01 até 2021-12
    mask_excl = (df_full[DATE_COL] < "2019-01-01") | (df_full[DATE_COL] > "2021-12-01")
    df_excl = df_full.loc[mask_excl].reset_index(drop=True)
    
    print(f"📊 Datasets carregados:")
    print(f"   • FULL: {len(df_full)} observações")
    print(f"   • EXCL: {len(df_excl)} observações (sem 2019-2021)")
    
    return df_full, df_excl


def split_train_test(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, 
                                                  np.ndarray, np.ndarray, List[str]]:
    """
    Split temporal 80/20.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset preparado
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    features = [col for col in df.columns if col not in [DATE_COL, TARGET]]
    X = df[features].values
    y = df[TARGET].values
    
    cut = int(len(df) * 0.8)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    
    return X_train, X_test, y_train, y_test, features


#%% ==================== MÉTRICAS ====================

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error em %."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional Accuracy: % de acertos na direção da variação."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if len(y_true) < 2:
        return np.nan
    
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    return ((true_diff > 0) == (pred_diff > 0)).mean() * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   model_name: str, scenario: str) -> Dict:
    """
    Calcula métricas de avaliação.
    
    Parameters
    ----------
    y_true : array-like
        Valores reais
    y_pred : array-like
        Valores preditos
    model_name : str
        Nome do modelo
    scenario : str
        Cenário (FULL ou EXCL)
    
    Returns
    -------
    dict
        Dicionário com métricas
    """
    return {
        "Cenario": scenario,
        "Model": model_name,
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DA": direction_accuracy(y_true, y_pred)
    }


#%% ==================== FUNÇÕES DE VISUALIZAÇÃO (ABNT) ====================

def format_abnt_axes(ax, xlabel: str = "", ylabel: str = "", show_legend: bool = True):
    """Formata eixos no padrão ABNT."""
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
    
    if show_legend:
        ax.legend(frameon=False, prop={"family": "Arial", "size": 9})


def plot_horizontal_bar(df: pd.DataFrame, value_col: str, title: str, 
                        filename: Path, xlabel: str):
    """Gráfico de barras horizontal (padrão ABNT)."""
    df_sorted = df.sort_values(value_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_sorted["Variavel"], df_sorted[value_col], color="#4A90E2")
    
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f"{width:.3f}", va="center", ha="left", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_real_vs_pred(dates, y_true, y_pred, title, outpath):
    """Gráfico Real vs Predito (padrão ABNT)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, y_true, label="Real", linewidth=2)
    ax.plot(dates, y_pred, label="Predito", linestyle="--", linewidth=2)
    
    format_abnt_axes(ax, xlabel="Data", ylabel="Taxa de inadimplência (%)", 
                    show_legend=True)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


#%% ==================== BLOCO 1: ANÁLISE DE COLINEARIDADE ====================
print("\n" + "="*80)
print("BLOCO 1: ANÁLISE DE COLINEARIDADE")
print("="*80)


def compute_correlation(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Calcula e salva matriz de correlação.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    tag : str
        Identificador (FULL ou EXCL)
    
    Returns
    -------
    pd.DataFrame
        Matriz de correlação
    """
    features = [col for col in df.columns if col not in [DATE_COL]]
    corr = df[features].corr()
    
    # Salvar CSV
    corr.to_csv(COLIN_DIR / f"correlation_{tag}.csv", float_format="%.4f")
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title(f"Matriz de Correlação — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(COLIN_DIR / f"heatmap_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✅ {tag}: Matriz de correlação salva")
    
    return corr


def compute_vif(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Calcula Variance Inflation Factor (VIF).
    
    VIF > 10 indica colinearidade problemática.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    tag : str
        Identificador (FULL ou EXCL)
    
    Returns
    -------
    pd.DataFrame
        DataFrame com VIF por variável
    """
    features = [col for col in df.columns if col not in [DATE_COL]]
    X = df[features].astype(float)
    
    # Padronizar antes de calcular VIF
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    vif_data = []
    for i, col in enumerate(features):
        vif_value = variance_inflation_factor(X_scaled, i)
        vif_data.append([col, vif_value])
    
    df_vif = pd.DataFrame(vif_data, columns=["Variavel", "VIF"])
    df_vif = df_vif.sort_values("VIF", ascending=False).reset_index(drop=True)
    
    # Salvar
    df_vif.to_csv(COLIN_DIR / f"vif_{tag}.csv", index=False, float_format="%.3f")
    
    print(f"   ✅ {tag}: VIF calculado e salvo")
    print(df_vif.to_string(index=False, float_format="%.3f"))
    
    return df_vif


def analyze_colinearity():
    """Executa análise completa de colinearidade."""
    print("\n📊 Analisando colinearidade...")
    print("-" * 80)
    
    df_full, df_excl = load_datasets()
    
    # Correlação
    print("\n1️⃣ Matriz de Correlação:")
    corr_full = compute_correlation(df_full, "FULL")
    corr_excl = compute_correlation(df_excl, "EXCL")
    
    # VIF
    print("\n2️⃣ Variance Inflation Factor (VIF):")
    vif_full = compute_vif(df_full, "FULL")
    vif_excl = compute_vif(df_excl, "EXCL")
    
    print(f"\n✅ Análise de colinearidade concluída")
    print(f"   Arquivos salvos em: {COLIN_DIR}")


#%% ==================== BLOCO 2: MODELAGEM EXPLORATÓRIA ====================
print("\n" + "="*80)
print("BLOCO 2: MODELAGEM EXPLORATÓRIA + DIAGNÓSTICOS")
print("="*80)


def make_mlp_model(input_dim: int) -> Model:
    """Cria modelo MLP."""
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    
    return model


def make_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """Cria modelo LSTM."""
    inp = Input(shape=input_shape)
    x = LSTM(64, activation="tanh")(inp)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    
    return model


def run_exploratory_analysis(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Executa modelagem exploratória completa.
    
    Modelos testados:
        1. Regressão Linear (baseline)
        2. SVR (kernel RBF)
        3. XGBoost
        4. MLP (Multilayer Perceptron)
        5. LSTM (Long Short-Term Memory)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset preparado
    tag : str
        Identificador do cenário (FULL ou EXCL)
    
    Returns
    -------
    pd.DataFrame
        Resultados consolidados
    """
    print(f"\n🔬 Análise Exploratória: {tag}")
    print("-" * 80)
    
    # Preparar dados
    X_train, X_test, y_train, y_test, features = split_train_test(df)
    dates_test = df[DATE_COL].values[int(len(df) * 0.8):]
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Escalonamento
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = []
    predictions = {}
    
    # 1. Regressão Linear
    print("\n   1️⃣ Regressão Linear...")
    lin = LinearRegression()
    lin.fit(X_train_s, y_train)
    y_pred_lin = lin.predict(X_test_s)
    predictions["Linear"] = y_pred_lin
    results.append(evaluate_model(y_test, y_pred_lin, "Linear", tag))
    
    # Diagnóstico: Coeficientes padronizados
    df_coef = pd.DataFrame({
        "Variavel": features,
        "Coef_padronizado": lin.coef_,
        "Abs_Coef": np.abs(lin.coef_),
        "Cenario": tag
    }).sort_values("Abs_Coef", ascending=False)
    df_coef.to_csv(RES_DIAG_DIR / f"linear_coeffs_{tag}.csv", 
                  index=False, float_format="%.6f")
    
    # 2. SVR
    print("   2️⃣ SVR...")
    svr = SVR(kernel="rbf", C=1.0, gamma="scale")
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)
    predictions["SVR"] = y_pred_svr
    results.append(evaluate_model(y_test, y_pred_svr, "SVR", tag))
    
    # 3. XGBoost
    print("   3️⃣ XGBoost...")
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    predictions["XGBoost"] = y_pred_xgb
    results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost", tag))
    
    # Diagnóstico: Importância das variáveis
    df_imp = pd.DataFrame({
        "Variavel": features,
        "Importancia": xgb.feature_importances_,
        "Importancia_pct": xgb.feature_importances_ / xgb.feature_importances_.sum() * 100,
        "Cenario": tag
    }).sort_values("Importancia", ascending=False)
    df_imp.to_csv(RES_DIAG_DIR / f"xgb_importance_{tag}.csv",
                 index=False, float_format="%.6f")
    
    # 4. MLP
    print("   4️⃣ MLP...")
    mlp = make_mlp_model(X_train_s.shape[1])
    es_mlp = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    mlp.fit(X_train_s, y_train, validation_split=0.2, epochs=200, 
           batch_size=8, verbose=0, callbacks=[es_mlp])
    y_pred_mlp = mlp.predict(X_test_s, verbose=0).flatten()
    predictions["MLP"] = y_pred_mlp
    results.append(evaluate_model(y_test, y_pred_mlp, "MLP", tag))
    
    # 5. LSTM
    print("   5️⃣ LSTM...")
    X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_r = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
    
    lstm = make_lstm_model((1, X_train_s.shape[1]))
    es_lstm = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    lstm.fit(X_train_r, y_train, validation_split=0.2, epochs=200,
            batch_size=8, verbose=0, callbacks=[es_lstm])
    y_pred_lstm = lstm.predict(X_test_r, verbose=0).flatten()
    predictions["LSTM"] = y_pred_lstm
    results.append(evaluate_model(y_test, y_pred_lstm, "LSTM", tag))
    
    # Consolidar resultados
    df_results = pd.DataFrame(results).sort_values(by="MSE").reset_index(drop=True)
    df_results.to_csv(RES_DIAG_DIR / f"results_{tag}_exploratory.csv",
                     index=False, float_format="%.4f")
    
    print(f"\n📊 Resultados ({tag}):")
    print(df_results.to_string(index=False, float_format="%.4f"))
    
    best = df_results.iloc[0]["Model"]
    print(f"\n🏆 Melhor modelo: {best}")
    
    # Gráficos Real vs Predito (apenas Linear e SVR)
    plot_real_vs_pred(dates_test, y_test, predictions["Linear"],
                     f"{tag} - Linear", 
                     RES_DIAG_DIR / f"{tag}_Linear_exploratory.png")
    
    plot_real_vs_pred(dates_test, y_test, predictions["SVR"],
                     f"{tag} - SVR",
                     RES_DIAG_DIR / f"{tag}_SVR_exploratory.png")
    
    # Gráficos de diagnóstico
    plot_horizontal_bar(df_coef.head(10), "Coef_padronizado",
                       f"Coeficientes Padronizados — {tag}",
                       PLOT_DIAG_DIR / f"linear_coef_{tag}.png",
                       "Coeficiente Padronizado")
    
    plot_horizontal_bar(df_imp.head(10), "Importancia_pct",
                       f"Importância das Variáveis — {tag}",
                       PLOT_DIAG_DIR / f"xgb_import_{tag}.png",
                       "Importância (%)")
    
    return df_results


def run_exploratory_modeling():
    """Executa modelagem exploratória para ambos os cenários."""
    print("\n🔬 Iniciando modelagem exploratória...")
    print("-" * 80)
    
    df_full, df_excl = load_datasets()
    
    results_full = run_exploratory_analysis(df_full, "FULL")
    results_excl = run_exploratory_analysis(df_excl, "EXCL")
    
    # Consolidar
    df_all = pd.concat([results_full, results_excl], ignore_index=True)
    df_all.to_csv(RES_DIAG_DIR / "results_exploratory_consolidated.csv",
                 index=False, float_format="%.4f")
    
    print(f"\n✅ Modelagem exploratória concluída")
    print(f"   Resultados salvos em: {RES_DIAG_DIR}")


#%% ==================== BLOCO 3: MODELAGEM OTIMIZADA ====================
print("\n" + "="*80)
print("BLOCO 3: MODELAGEM OTIMIZADA (FEATURES SELECIONADAS)")
print("="*80)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona features finais baseado em análise de colinearidade.
    
    Features selecionadas (VIF < 10):
        - inadimpl_cartao_total_lag1
        - ibcbr_dessaz
        - selic_mensal
        - comprometimento_renda
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo
    
    Returns
    -------
    pd.DataFrame
        Dataset com features selecionadas
    """
    selected = [
        DATE_COL,
        TARGET,
        "inadimpl_cartao_total_lag1",
        "ibcbr_dessaz",
        "selic_mensal",
        "comprometimento_renda",
	"ipca_mensal"
    ]
    
    cols_exist = [col for col in selected if col in df.columns]
    
    return df[cols_exist].copy()


def run_optimized_analysis(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Executa modelagem otimizada com features selecionadas.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo
    tag : str
        Identificador do cenário
    
    Returns
    -------
    pd.DataFrame
        Resultados consolidados
    """
    print(f"\n🎯 Modelagem Otimizada: {tag}")
    print("-" * 80)
    
    # Selecionar features
    df = select_features(df)
    print(f"   Features selecionadas: {len(df.columns) - 2}")
    
    # Preparar dados
    X_train, X_test, y_train, y_test, features = split_train_test(df)
    dates_test = df[DATE_COL].values[int(len(df) * 0.8):]
    
    # Escalonamento
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = []
    predictions = {}
    
    # 1. Regressão Linear
    print("   1️⃣ Linear Regression...")
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    predictions["Linear"] = y_pred_lin
    results.append(evaluate_model(y_test, y_pred_lin, "Linear", tag))
    
    # Diagnóstico
    lin_std = LinearRegression()
    lin_std.fit(X_train_s, y_train)
    df_coef = pd.DataFrame({
        "Variavel": features,
        "Coef_padronizado": lin_std.coef_,
        "Abs_Coef": np.abs(lin_std.coef_),
        "Cenario": tag
    }).sort_values("Abs_Coef", ascending=False)
    df_coef.to_csv(DIAG_FINAL_DIR / f"linear_coeffs_{tag}.csv",
                  index=False, float_format="%.6f")
    
    # 2. SVR
    print("   2️⃣ SVR...")
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)
    predictions["SVR"] = y_pred_svr
    results.append(evaluate_model(y_test, y_pred_svr, "SVR", tag))
    
    # 3. XGBoost
    print("   3️⃣ XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    predictions["XGBoost"] = y_pred_xgb
    results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost", tag))
    
    # Diagnóstico
    df_imp = pd.DataFrame({
        "Variavel": features,
        "Importancia": xgb.feature_importances_,
        "Importancia_pct": xgb.feature_importances_ / xgb.feature_importances_.sum() * 100,
        "Cenario": tag
    }).sort_values("Importancia", ascending=False)
    df_imp.to_csv(DIAG_FINAL_DIR / f"xgb_importance_{tag}.csv",
                 index=False, float_format="%.3f")
    
    # 4. MLP
    print("   4️⃣ MLP...")
    mlp = make_mlp_model(X_train_s.shape[1])
    es_mlp = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    mlp.fit(X_train_s, y_train, validation_split=0.2, epochs=400,
           batch_size=8, verbose=0, callbacks=[es_mlp])
    y_pred_mlp = mlp.predict(X_test_s, verbose=0).flatten()
    predictions["MLP"] = y_pred_mlp
    results.append(evaluate_model(y_test, y_pred_mlp, "MLP", tag))
    
    # 5. LSTM
    print("   5️⃣ LSTM...")
    X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_r = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
    
    lstm = make_lstm_model((1, X_train_s.shape[1]))
    es_lstm = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    lstm.fit(X_train_r, y_train, validation_split=0.2, epochs=400,
            batch_size=8, verbose=0, callbacks=[es_lstm])
    y_pred_lstm = lstm.predict(X_test_r, verbose=0).flatten()
    predictions["LSTM"] = y_pred_lstm
    results.append(evaluate_model(y_test, y_pred_lstm, "LSTM", tag))
    
    # Consolidar
    df_results = pd.DataFrame(results).sort_values(by="MSE").reset_index(drop=True)
    df_results.to_csv(RES_FINAL_DIR / f"results_{tag}_final.csv",
                     index=False, float_format="%.4f")
    
    print(f"\n📊 Resultados Finais ({tag}):")
    print(df_results.to_string(index=False, float_format="%.4f"))
    
    best = df_results.iloc[0]["Model"]
    print(f"\n🏆 Melhor modelo: {best}")
    
    # Gráficos
    plot_real_vs_pred(dates_test, y_test, predictions["Linear"],
                     f"{tag} - Linear Final",
                     RES_FINAL_DIR / f"{tag}_Linear_final.png")
    
    plot_real_vs_pred(dates_test, y_test, predictions["SVR"],
                     f"{tag} - SVR Final",
                     RES_FINAL_DIR / f"{tag}_SVR_final.png")
    
    return df_results


def run_optimized_modeling():
    """Executa modelagem otimizada para ambos os cenários."""
    print("\n🎯 Iniciando modelagem otimizada...")
    print("-" * 80)
    
    df_full, df_excl = load_datasets()
    
    results_full = run_optimized_analysis(df_full, "FULL")
    results_excl = run_optimized_analysis(df_excl, "EXCL")
    
    # Consolidar
    df_all = pd.concat([results_full, results_excl], ignore_index=True)
    df_all.to_csv(RES_FINAL_DIR / "results_FULL_EXCL_consolidated.csv",
                 index=False, float_format="%.4f")
    
    print(f"\n✅ Modelagem otimizada concluída")
    print(f"   Resultados salvos em: {RES_FINAL_DIR}")


#%% ==================== EXECUÇÃO PRINCIPAL ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TCC - PREVISÃO DA INADIMPLÊNCIA DE CARTÕES DE CRÉDITO NO BRASIL")
    print("PIPELINE DE ANÁLISES E MODELAGEM")
    print("="*80)
    
    # Bloco 1: Colinearidade
    try:
        analyze_colinearity()
    except Exception as e:
        print(f"\n❌ ERRO na análise de colinearidade: {e}")
        import traceback
        traceback.print_exc()
    
    # Bloco 2: Modelagem Exploratória
    try:
        run_exploratory_modeling()
    except Exception as e:
        print(f"\n❌ ERRO na modelagem exploratória: {e}")
        import traceback
        traceback.print_exc()
    
    # Bloco 3: Modelagem Otimizada
    try:
        run_optimized_modeling()
    except Exception as e:
        print(f"\n❌ ERRO na modelagem otimizada: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumo final
    print("\n" + "="*80)
    print("✅ PIPELINE DE ANÁLISES CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Resultados gerados:")
    print(f"   1. Análise de Colinearidade: {COLIN_DIR}")
    print(f"   2. Modelagem Exploratória: {RES_DIAG_DIR}")
    print(f"   3. Modelagem Otimizada: {RES_FINAL_DIR}")
    print(f"   4. Gráficos de Diagnóstico: {PLOT_DIAG_DIR}")
    print("\n📊 Arquivos CSV disponíveis:")
    print("   • correlation_FULL/EXCL.csv")
    print("   • vif_FULL/EXCL.csv")
    print("   • linear_coeffs_FULL/EXCL.csv")
    print("   • xgb_importance_FULL/EXCL.csv")
    print("   • results_FULL/EXCL_final.csv")
    print("   • results_FULL_EXCL_consolidated.csv")
    print("\n🎨 Visualizações disponíveis:")
    print("   • Heatmaps de correlação")
    print("   • Gráficos de coeficientes/importâncias")
    print("   • Gráficos Real vs Predito")
    print("="*80 + "\n")
