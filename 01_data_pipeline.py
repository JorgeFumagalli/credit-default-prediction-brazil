#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
01_data_pipeline.py
TCC — Previsão da Inadimplência de Cartões de Crédito no Brasil

SCRIPT 1: EXTRAÇÃO, PREPARAÇÃO E DIAGNÓSTICOS ESTATÍSTICOS
==============================================================================

Este script consolida as etapas iniciais do pipeline:
    1. Extração das séries temporais do BCB/SGS
    2. Consolidação da base macroeconômica
    3. Preparação dos datasets FULL e EXCL
    4. Diagnósticos estatísticos da etapa de preparação

Saídas principais:
    - ./data/dados_consolidados_macro_credito.parquet
    - ./prepared/prepared_FULL.parquet
    - ./prepared/prepared_EXCL.parquet
    - ./results_preparation/*

Observações metodológicas:
    - As variáveis explicativas entram em nível (sem lags generalizados)
    - É criada apenas TARGET_lag1
    - Não há imputação por forward fill
    - O cenário EXCL remove 2019-01-01 a 2021-12-01
    - O stepwise da preparação fica desativado por padrão
==============================================================================
"""

# %% ==================== IMPORTAÇÕES ====================
import os
import warnings
warnings.filterwarnings("ignore")

from functools import reduce
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from scipy.stats import boxcox, norm, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# Dependências opcionais
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except Exception:
    HAS_PINGOUIN = False

try:
    from statstests.tests import shapiro_francia
    HAS_SHAPIRO_FRANCIA = True
except Exception:
    HAS_SHAPIRO_FRANCIA = False

_STEPWISE_IMPORTED = False
try:
    from statstests.process import stepwise
    _STEPWISE_IMPORTED = True
except Exception:
    _STEPWISE_IMPORTED = False

ENABLE_STEPWISE_PREP = os.getenv("ENABLE_STEPWISE_PREP", "0") == "1" and _STEPWISE_IMPORTED

# %% ==================== CONFIGURAÇÕES ====================
DATA_DIR = Path("./data")
PREP_DIR = Path("./prepared")
RES_DIR = Path("./results_preparation")

for directory in [DATA_DIR, PREP_DIR, RES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

START_DATE = "2011-03-01"
END_DATE = "2025-07-01"
DATE_COL = "data"
TARGET = "inadimpl_cartao_total"

SERIES = {
    "selic_mensal": 4390,
    "ibcbr_dessaz": 24364,
    "ibcbr_sem_ajuste": 24363,
    "inadimpl_cartao_total": 25464,
    "ipca_mensal": 433,
    "comprometimento_renda": 29034,
    "endividamento_familias": 29037,
}

FEATURES_RAW = [
    "selic_mensal",
    "ibcbr_dessaz",
    "ibcbr_sem_ajuste",
    "ipca_mensal",
    "comprometimento_renda",
    "endividamento_familias",
]

EXPECTED_REQUIRED_COLS = [DATE_COL, TARGET] + FEATURES_RAW
EXPECTED_RAW_ROWS = 173
EXPECTED_FULL_ROWS = 172
EXPECTED_EXCL_ROWS = 136
EXCL_START = "2019-01-01"
EXCL_END = "2021-12-01"
TOPK_SCATTER = 8
STEPWISE_P = 0.05

CONSOLIDATED_PATH = DATA_DIR / "dados_consolidados_macro_credito.parquet"
FULL_PATH = PREP_DIR / "prepared_FULL.parquet"
EXCL_PATH = PREP_DIR / "prepared_EXCL.parquet"
RUNTIME_NOTES = RES_DIR / "runtime_notes.txt"
SUMMARY_PATH = RES_DIR / "dataset_summary.txt"

SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (SGS-client; TCC Jorge Fumagalli)",
})

# %% ==================== UTILIDADES DE I/O ====================
def append_runtime_note(message: str):
    with RUNTIME_NOTES.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def safe_write_table(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        df.to_pickle(path)
        append_runtime_note(
            f"Fallback de escrita para pickle em '{path.name}' por indisponibilidade de engine parquet: {exc}"
        )


def safe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    parquet_error = None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        parquet_error = exc

    try:
        df = pd.read_pickle(path)
        append_runtime_note(
            f"Fallback de leitura para pickle em '{path.name}' porque a leitura parquet falhou: {parquet_error}"
        )
        return df
    except Exception as pickle_exc:
        raise RuntimeError(
            f"Não foi possível ler '{path}'. Falha parquet: {parquet_error}; falha pickle: {pickle_exc}"
        )

# %% ==================== ESTILO DOS GRÁFICOS ====================
def format_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "", legend: bool = False):
    if title:
        ax.set_title(title, fontfamily="DejaVu Sans", fontsize=12, color="black")

    ax.set_xlabel(xlabel, fontfamily="DejaVu Sans", fontsize=11, color="black")
    ax.set_ylabel(ylabel, fontfamily="DejaVu Sans", fontsize=11, color="black")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("DejaVu Sans")
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

    if legend:
        ax.legend(frameon=False, prop={"family": "DejaVu Sans", "size": 9})

# %% ==================== VALIDAÇÃO DA BASE ====================
def is_expected_consolidated_base(df: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    temp = df.copy()

    if DATE_COL not in temp.columns:
        reasons.append("coluna de data ausente")
        return False, reasons

    temp[DATE_COL] = pd.to_datetime(temp[DATE_COL])
    temp = temp.sort_values(DATE_COL).reset_index(drop=True)

    missing_cols = [c for c in EXPECTED_REQUIRED_COLS if c not in temp.columns]
    if missing_cols:
        reasons.append("colunas ausentes: " + ", ".join(missing_cols))

    if len(temp) != EXPECTED_RAW_ROWS:
        reasons.append(f"quantidade de linhas esperada {EXPECTED_RAW_ROWS}, encontrada {len(temp)}")

    dmin = temp[DATE_COL].min()
    dmax = temp[DATE_COL].max()
    if pd.isna(dmin) or pd.isna(dmax):
        reasons.append("datas inválidas na base")
    else:
        if dmin != pd.to_datetime(START_DATE):
            reasons.append(f"data inicial esperada {START_DATE}, encontrada {dmin.date()}")
        if dmax != pd.to_datetime(END_DATE):
            reasons.append(f"data final esperada {END_DATE}, encontrada {dmax.date()}")

    return len(reasons) == 0, reasons

# %% ==================== FUNÇÕES DE EXTRAÇÃO ====================
def _fmt(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%d/%m/%Y")


def _fetch_sgs(host: str, codigo: int, start: str, end: str, timeout: int = 30) -> requests.Response:
    url = f"{host}/dados/serie/bcdata.sgs.{codigo}/dados"
    return SESSION.get(
        url,
        params={
            "formato": "json",
            "dataInicial": _fmt(start),
            "dataFinal": _fmt(end),
        },
        timeout=timeout,
    )


def get_sgs(codigo: int, start: str = START_DATE, end: str = END_DATE, timeout: int = 30) -> pd.DataFrame:
    hosts = [
        "https://api.bcb.gov.br",
        "https://dadosabertos.bcb.gov.br",
    ]

    for host in hosts:
        try:
            response = _fetch_sgs(host, codigo, start, end, timeout)
            ctype = response.headers.get("Content-Type", "").lower()

            if response.status_code == 200 and "json" in ctype:
                data = response.json()
                df = pd.DataFrame(data)

                if df.empty:
                    return pd.DataFrame(columns=[DATE_COL, "valor"])

                df["valor"] = pd.to_numeric(df["valor"].astype(str).str.replace(",", "."), errors="coerce")
                df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True)
                return df
        except Exception as exc:
            append_runtime_note(f"Falha ao consultar host {host} para série {codigo}: {exc}")
            continue

    raise RuntimeError(f"Falha ao baixar a série SGS {codigo}.")


def to_month_start(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = out[DATE_COL].dt.to_period("M").dt.start_time
    return out


def extract_all_series() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("BLOCO 1: EXTRAÇÃO DAS SÉRIES DO BCB/SGS")
    print("=" * 80)

    force_download = os.getenv("FORCE_BCB_DOWNLOAD", "0") == "1"
    cached_valid = False
    cached_df = None

    if CONSOLIDATED_PATH.exists() and not force_download:
        try:
            cached_df = safe_read_table(CONSOLIDATED_PATH)
            cached_valid, reasons = is_expected_consolidated_base(cached_df)
            if cached_valid:
                print("Base consolidada local válida encontrada. Pulando download do BCB/SGS.")
                append_runtime_note("Base consolidada local validada com sucesso; download do BCB/SGS foi pulado.")
                return cached_df

            print("⚠️ Base consolidada local encontrada, mas fora do padrão do TCC. Forçando nova extração.")
            append_runtime_note("Base local rejeitada por inconsistência com o TCC: " + " | ".join(reasons))
        except Exception as exc:
            print(f"⚠️ Falha ao validar base consolidada local: {exc}. Tentando nova extração.")
            append_runtime_note(f"Falha ao validar base consolidada local: {exc}")

    dataframes = []
    for name, code in SERIES.items():
        try:
            print(f"Baixando: {name} (SGS {code})...", end=" ")
            df = get_sgs(code, START_DATE, END_DATE)
            df = to_month_start(df).rename(columns={"valor": name})
            df = df[[DATE_COL, name]]
            dataframes.append(df)
            print(f"✅ {len(df)} observações")
        except Exception as exc:
            print(f"❌ ERRO: {exc}")

    if len(dataframes) == len(SERIES):
        base = reduce(lambda l, r: pd.merge(l, r, on=DATE_COL, how="outer"), dataframes)
        base = base.sort_values(DATE_COL).reset_index(drop=True)
        safe_write_table(base, CONSOLIDATED_PATH)
        print(f"\n✅ Base consolidada salva em: {CONSOLIDATED_PATH}")
        ok, reasons = is_expected_consolidated_base(base)
        if not ok:
            raise RuntimeError(
                "A nova base consolidada foi criada, mas não está coerente com o padrão do TCC: " + " | ".join(reasons)
            )
        return base

    if cached_valid and cached_df is not None:
        print("\n⚠️ Nem todas as séries puderam ser baixadas. Usando base consolidada local válida já existente.")
        append_runtime_note("Nem todas as séries foram baixadas; pipeline seguiu com base local válida já existente.")
        return cached_df

    raise RuntimeError(
        "Não foi possível montar uma base coerente com o TCC. Verifique a conectividade com o BCB e remova bases antigas inconsistentes."
    )

# %% ==================== PREPARAÇÃO DOS DADOS ====================
def ensure_month_start(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return dates.values.astype("datetime64[M]").astype("datetime64[ns]")


def load_and_select() -> pd.DataFrame:
    df = safe_read_table(CONSOLIDATED_PATH).copy()
    df[DATE_COL] = ensure_month_start(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' não encontrada na base.")

    feats = [c for c in FEATURES_RAW if c in df.columns]
    missing = [c for c in FEATURES_RAW if c not in df.columns]

    if missing:
        (RES_DIR / "missing_features.txt").write_text(
            "Features configuradas ausentes na base:\n" + "\n".join(missing),
            encoding="utf-8",
        )

    selected = [DATE_COL, TARGET] + feats
    df = df[selected].copy()

    missingness = df.isna().mean().sort_values(ascending=False)
    (RES_DIR / "missingness_report.txt").write_text(
        "Proporção de valores ausentes por coluna:\n\n" + missingness.to_string(),
        encoding="utf-8",
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[f"{TARGET}_lag1"] = out[TARGET].shift(1)
    out = out.dropna().reset_index(drop=True)
    return out


def make_scenarios(df_features: pd.DataFrame):
    df_full = df_features.copy()
    mask_drop = (
        (df_full[DATE_COL] >= pd.to_datetime(EXCL_START))
        & (df_full[DATE_COL] <= pd.to_datetime(EXCL_END))
    )
    df_excl = df_full.loc[~mask_drop].reset_index(drop=True)
    return df_full, df_excl

# %% ==================== AUXILIARES DOS DIAGNÓSTICOS ====================
def _safe_boxcox(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    shift = 0.0
    xmin = np.nanmin(x)
    if xmin <= 0:
        shift = abs(xmin) + 1e-6
        x = x + shift
    x_bc, lam = boxcox(x)
    return x_bc, float(lam), float(shift)


def _heatmap_corr(corr: pd.DataFrame, outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    format_axes(ax, title=title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_matrix(df_small: pd.DataFrame, outpath: Path, title: str):
    from pandas.plotting import scatter_matrix

    fig = plt.figure(figsize=(14, 14))
    axes = scatter_matrix(df_small, alpha=0.7, diagonal="hist", figsize=(14, 14))
    for ax in axes.flatten():
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily("DejaVu Sans")
            lab.set_fontsize(7)
    plt.suptitle(title, fontfamily="DejaVu Sans", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _residual_plots(resid: np.ndarray, out_hist: Path, out_qq: Path, title_prefix: str):
    resid = np.asarray(resid, dtype=float)
    mu, sigma = norm.fit(resid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resid, bins=20, density=True, alpha=0.6)
    x = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma), linewidth=2)
    format_axes(ax, xlabel="Resíduos", ylabel="Densidade", title=f"{title_prefix} — Histograma dos resíduos e Normal teórica")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    sm.qqplot(resid, line="45", ax=ax)
    format_axes(ax, xlabel="Quantis teóricos", ylabel="Quantis amostrais", title=f"{title_prefix} — QQ-plot dos resíduos")
    plt.tight_layout()
    plt.savefig(out_qq, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% ==================== DIAGNÓSTICOS ESTATÍSTICOS ====================
def run_diagnostics(df: pd.DataFrame, tag: str):
    print(f"\n=== Diagnósticos estatísticos ({tag}) ===")
    prefix = f"{tag}_"
    cols_num = [c for c in df.columns if c != DATE_COL]

    desc = df[cols_num].describe().T
    desc.to_csv(RES_DIR / f"{prefix}describe.csv", encoding="utf-8")

    corr = df[cols_num].corr(method="pearson")
    corr.to_csv(RES_DIR / f"{prefix}corr_matrix.csv", encoding="utf-8")
    _heatmap_corr(corr, RES_DIR / f"{prefix}corr_heatmap.png", f"{tag} — Matriz de correlação (Pearson)")

    if HAS_PINGOUIN:
        try:
            rcorr = pg.rcorr(df[cols_num], method="pearson", upper="pval", decimals=6)
            (RES_DIR / f"{prefix}corr_rcorr_pingouin.txt").write_text(str(rcorr), encoding="utf-8")
        except Exception as exc:
            (RES_DIR / f"{prefix}corr_rcorr_pingouin.txt").write_text(
                f"Falha ao calcular rcorr com pingouin: {exc}", encoding="utf-8"
            )
    else:
        (RES_DIR / f"{prefix}corr_rcorr_pingouin.txt").write_text(
            "pingouin não disponível no ambiente.", encoding="utf-8"
        )

    target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    top_feats = target_corr.head(min(TOPK_SCATTER, len(target_corr))).index.tolist()
    if top_feats:
        _scatter_matrix(df[top_feats + [TARGET]].copy(), RES_DIR / f"{prefix}scatter_matrix_pairs.png", f"{tag} — Scatter-matrix")

    feats = [c for c in cols_num if c != TARGET]
    X = df[feats].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xv = sm.add_constant(Xs)

    vif = pd.DataFrame({
        "Variavel": ["const"] + feats,
        "VIF": [variance_inflation_factor(Xv, i) for i in range(Xv.shape[1])],
    })
    vif["Tolerancia"] = 1.0 / vif["VIF"].replace(0, np.nan)
    vif.to_csv(RES_DIR / f"{prefix}vif_table.csv", index=False, encoding="utf-8")

    cond_number = np.linalg.cond(Xv)
    y = df[TARGET].values
    lin = LinearRegression()
    lin.fit(Xs, y)
    resid = y - lin.predict(Xs)

    lines = [
        f"=== NORMALIDADE DOS RESÍDUOS ({tag}) — baseline linear ===\n\n",
        f"N observações: {len(resid)}\n",
        f"Condition number (com constante): {cond_number:.4f}\n\n",
    ]

    try:
        sw_stat, sw_p = shapiro(resid)
        lines.append(f"Shapiro-Wilk: W={sw_stat:.6f}, p-value={sw_p:.6f}\n")
    except Exception as exc:
        lines.append(f"Shapiro-Wilk: falhou ({exc})\n")

    if HAS_SHAPIRO_FRANCIA:
        try:
            sf_result = shapiro_francia(resid)
            sf_values = list(sf_result.items())
            w_value = sf_values[1][1]
            p_value = sf_values[3][1]
            lines.append(f"Shapiro-Francia: W={w_value:.6f}, p-value={p_value:.6f}\n")
        except Exception as exc:
            lines.append(f"Shapiro-Francia: falhou ({exc})\n")
    else:
        lines.append("Shapiro-Francia: pacote 'statstests' não disponível.\n")

    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    lines.append(f"Jarque-Bera: JB={jb_stat:.6f}, p-value={jb_p:.6f}, skew={skew:.6f}, kurt={kurt:.6f}\n")
    (RES_DIR / f"{prefix}residuals_normality_tests.txt").write_text("".join(lines), encoding="utf-8")

    _residual_plots(
        resid,
        RES_DIR / f"{prefix}residuals_hist.png",
        RES_DIR / f"{prefix}residuals_qqplot.png",
        title_prefix=tag,
    )

    try:
        _, lam, shift = _safe_boxcox(y)
        (RES_DIR / f"{prefix}boxcox_target_info.txt").write_text(
            "=== BOX-COX (TARGET) ===\n"
            f"Lambda estimado: {lam:.6f}\n"
            f"Shift aplicado: {shift:.6f}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        (RES_DIR / f"{prefix}boxcox_target_info.txt").write_text(f"Box-Cox falhou: {exc}", encoding="utf-8")

    if ENABLE_STEPWISE_PREP:
        try:
            df_ols = df[cols_num].copy()
            feats_formula = [c for c in cols_num if c != TARGET]
            formula = TARGET + " ~ " + " + ".join(feats_formula)
            modelo = sm.OLS.from_formula(formula, df_ols).fit()
            modelo_step = stepwise(modelo, pvalue_limit=STEPWISE_P)
            params = list(modelo_step.params.index)
            selected = [p for p in params if p not in ["Intercept", "const"]]
            (RES_DIR / f"{prefix}stepwise_selected_features.txt").write_text(
                "=== FEATURES SELECIONADAS (STEPWISE) ===\n"
                f"p-value limite: {STEPWISE_P}\n\n" + ("\n".join(selected) if selected else "(nenhuma feature selecionada)"),
                encoding="utf-8",
            )
        except Exception as exc:
            (RES_DIR / f"{prefix}stepwise_selected_features.txt").write_text(
                f"Stepwise falhou: {exc}", encoding="utf-8"
            )
    else:
        (RES_DIR / f"{prefix}stepwise_selected_features.txt").write_text(
            "Stepwise da preparação desativado por padrão. Defina ENABLE_STEPWISE_PREP=1 para ativar.",
            encoding="utf-8",
        )

    print(f"OK: diagnósticos de {tag} salvos em {RES_DIR.resolve()}")

# %% ==================== EXECUÇÃO PRINCIPAL ====================
def main():
    RUNTIME_NOTES.write_text("", encoding="utf-8")
    append_runtime_note(f"HAS_PINGOUIN={HAS_PINGOUIN}")
    append_runtime_note(f"HAS_SHAPIRO_FRANCIA={HAS_SHAPIRO_FRANCIA}")
    append_runtime_note(f"ENABLE_STEPWISE_PREP={ENABLE_STEPWISE_PREP}")

    base = extract_all_series()
    print(f"Período bruto: {pd.to_datetime(base[DATE_COL]).min().date()} até {pd.to_datetime(base[DATE_COL]).max().date()}")
    print(f"Observações brutas: {len(base)}")

    df_raw = load_and_select()
    df_feat = build_features(df_raw)
    df_full, df_excl = make_scenarios(df_feat)

    if len(df_full) != EXPECTED_FULL_ROWS or len(df_excl) != EXPECTED_EXCL_ROWS:
        raise RuntimeError(
            f"Base preparada incoerente com o TCC. Esperado FULL={EXPECTED_FULL_ROWS} e EXCL={EXPECTED_EXCL_ROWS}; obtido FULL={len(df_full)} e EXCL={len(df_excl)}."
        )

    safe_write_table(df_full, FULL_PATH)
    safe_write_table(df_excl, EXCL_PATH)

    SUMMARY_PATH.write_text(
        "=== RESUMO DOS DATASETS ===\n"
        f"Base consolidada: {len(base)} observações\n"
        f"FULL: {len(df_full)} observações\n"
        f"EXCL: {len(df_excl)} observações\n"
        f"Período FULL: {df_full[DATE_COL].min().date()} até {df_full[DATE_COL].max().date()}\n"
        f"Período EXCL: {df_excl[DATE_COL].min().date()} até {df_excl[DATE_COL].max().date()}\n",
        encoding="utf-8",
    )

    print(f"Observações FULL: {len(df_full)}")
    print(f"Observações EXCL (sem 2019–2021): {len(df_excl)}")
    print(f"Datasets salvos em: {PREP_DIR.resolve()}")

    run_diagnostics(df_full, "FULL")
    run_diagnostics(df_excl, "EXCL")

    print("\n✅ Script 1 concluído com sucesso.")

if __name__ == "__main__":
    main()
