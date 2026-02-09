#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
01_data_pipeline.py
TCC — Previsão da Inadimplência de Cartões de Crédito no Brasil

SCRIPT 1: EXTRAÇÃO E PREPARAÇÃO DE DADOS
==============================================================================

Este script consolida as etapas iniciais do pipeline:
    1. Extração de séries temporais do BCB/SGS
    2. Preparação dos dados para modelagem

Autor: Jorge Fumagalli
Data: Janeiro 2026
==============================================================================
"""

#%% ==================== IMPORTS ====================
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce
from typing import Tuple

#%% ==================== CONFIGURAÇÕES ====================
# Diretórios
DATA_DIR = Path("./data")
PREP_DIR = Path("./prepared")

# Criar diretórios se não existirem
DATA_DIR.mkdir(parents=True, exist_ok=True)
PREP_DIR.mkdir(parents=True, exist_ok=True)

# Período de análise
START_DATE = "2015-01-01"
END_DATE   = "2025-07-01"

# Séries do Banco Central (SGS)
SERIES = {
    "selic_mensal":           4390,   # Taxa Selic mensal
    "ibcbr_dessaz":           24364,  # IBC-Br dessazonalizado
    "inadimpl_cartao_total":  25464,  # Inadimplência de cartão (%)
    "ipca_mensal":            433,    # IPCA mensal
    "comprometimento_renda":  29034,  # Comprometimento de renda das famílias
}

# Constantes
TARGET     = "inadimpl_cartao_total"
DATE_COL   = "data"
MIN_NONNA_RATIO = 0.70  # Mínimo de 70% não-nulos


#%% ==================== BLOCO 1: EXTRAÇÃO DE DADOS ====================
print("="*80)
print("BLOCO 1: EXTRAÇÃO DE SÉRIES DO BANCO CENTRAL (BCB/SGS)")
print("="*80)

# Sessão HTTP configurada
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (SGS-client; TCC Jorge Fumagalli)"
})


def format_date(date_str: str) -> str:
    """Converte data para formato dd/mm/yyyy aceito pela API do BCB."""
    return pd.to_datetime(date_str).strftime("%d/%m/%Y")


def fetch_sgs_serie(host: str, codigo: int, start: str, end: str, 
                    timeout: int = 60) -> requests.Response:
    """
    Faz requisição HTTP para a API do SGS/BCB.
    
    Parameters
    ----------
    host : str
        URL base da API
    codigo : int
        Código da série temporal
    start : str
        Data inicial
    end : str
        Data final
    timeout : int
        Timeout da requisição em segundos
    
    Returns
    -------
    requests.Response
        Resposta da requisição HTTP
    """
    url = f"{host}/dados/serie/bcdata.sgs.{codigo}/dados"
    response = SESSION.get(
        url,
        params={
            "formato": "json", 
            "dataInicial": format_date(start), 
            "dataFinal": format_date(end)
        },
        timeout=timeout
    )
    return response


def get_sgs(codigo: int, start: str = START_DATE, end: str = END_DATE,
            timeout: int = 60) -> pd.DataFrame:
    """
    Baixa série temporal do SGS/BCB com fallback para hosts alternativos.
    
    Parameters
    ----------
    codigo : int
        Código da série no SGS
    start : str
        Data inicial
    end : str
        Data final
    timeout : int
        Timeout em segundos
    
    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ['data', 'valor']
    
    Raises
    ------
    RuntimeError
        Se a série não puder ser baixada de nenhum host
    """
    hosts = [
        "https://api.bcb.gov.br",
        "https://dadosabertos.bcb.gov.br"
    ]
    
    for host in hosts:
        try:
            response = fetch_sgs_serie(host, codigo, start, end, timeout)
            
            if response.status_code == 200 and "json" in response.headers.get("Content-Type", "").lower():
                data = response.json()
                df = pd.DataFrame(data)
                
                if df.empty:
                    return pd.DataFrame(columns=["data", "valor"])
                
                # Converter valor para numérico (trata vírgula como decimal)
                df["valor"] = pd.to_numeric(
                    df["valor"].astype(str).str.replace(",", "."), 
                    errors="coerce"
                )
                
                # Converter data
                df["data"] = pd.to_datetime(df["data"], dayfirst=True)
                
                return df
                
        except Exception as e:
            print(f"   ⚠️ Tentativa falhou em {host}: {e}")
            continue
    
    raise RuntimeError(f"❌ Falha ao baixar série SGS {codigo} de todos os hosts")


def to_month_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte datas para o início do mês.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com coluna 'data'
    
    Returns
    -------
    pd.DataFrame
        DataFrame com datas convertidas para início do mês
    """
    df = df.copy()
    df["data"] = df["data"].dt.to_period("M").dt.start_time
    return df


def extract_all_series() -> pd.DataFrame:
    """
    Extrai todas as séries configuradas e consolida em um único DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame consolidado com todas as séries
    """
    print("\n📊 Baixando séries temporais...")
    print("-" * 80)
    
    dataframes = []
    
    for name, code in SERIES.items():
        try:
            print(f"   Baixando: {name} (código SGS {code})...", end=" ")
            
            df = get_sgs(code, START_DATE, END_DATE)
            df = to_month_start(df)
            df = df.rename(columns={"valor": name})
            df = df[["data", name]]
            
            dataframes.append(df)
            
            print(f"✅ {len(df)} observações")
            
        except Exception as e:
            print(f"❌ ERRO: {e}")
    
    # Consolidar todas as séries
    print("\n🔗 Consolidando séries...")
    base = reduce(
        lambda left, right: pd.merge(left, right, on="data", how="outer"), 
        dataframes
    )
    
    base = base.sort_values("data").reset_index(drop=True)
    
    # Salvar
    output_path = DATA_DIR / "dados_consolidados_macro_credito.parquet"
    base.to_parquet(output_path, index=False)
    
    print(f"✅ Dataset consolidado salvo: {output_path}")
    print(f"   • Período: {base['data'].min().date()} a {base['data'].max().date()}")
    print(f"   • Observações: {len(base)}")
    print(f"   • Variáveis: {len(base.columns) - 1}")
    
    return base


#%% ==================== BLOCO 2: PREPARAÇÃO DOS DADOS ====================
print("\n" + "="*80)
print("BLOCO 2: PREPARAÇÃO DOS DADOS PARA MODELAGEM")
print("="*80)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e organiza o DataFrame.
    
    - Converte coluna de data
    - Ordena cronologicamente
    - Filtra colunas com boa completude (>= 70% não-nulos)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto
    
    Returns
    -------
    pd.DataFrame
        DataFrame limpo
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    # Selecionar colunas com boa completude
    keep_cols = [DATE_COL] + [
        col for col in df.columns
        if col != DATE_COL and df[col].notna().mean() >= MIN_NONNA_RATIO
    ]
    
    df = df[keep_cols]
    
    return df


def finalize_dataset(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Remove linhas com valores ausentes e organiza colunas finais.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    feature_cols : list
        Lista de colunas de features
    
    Returns
    -------
    pd.DataFrame
        DataFrame finalizado sem valores ausentes
    """
    final = df.dropna(subset=[TARGET] + feature_cols).copy()
    final = final[[DATE_COL, TARGET] + feature_cols]
    
    return final


def prepare_full_dataset() -> pd.DataFrame:
    """
    Prepara o dataset completo para modelagem.
    
    - Carrega dados consolidados
    - Cria lag de 1 mês da variável target
    - Remove linhas com valores ausentes
    - Salva dataset preparado
    
    Returns
    -------
    pd.DataFrame
        Dataset preparado para modelagem
    
    Strategy
    --------
    Para evitar data leakage, usamos:
        • X_t: variáveis macroeconômicas do mês t
        • y_{t-1}: inadimplência do mês anterior (TARGET_lag1)
        • TARGET_t: inadimplência a ser prevista
    """
    print("\n📋 Preparando dataset FULL...")
    print("-" * 80)
    
    # Carregar dados
    input_file = DATA_DIR / "dados_consolidados_macro_credito.parquet"
    
    if not input_file.exists():
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {input_file}")
    
    base = pd.read_parquet(input_file)
    base = clean_dataframe(base)
    
    # Identificar preditores (todas as variáveis numéricas exceto target)
    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [col for col in numeric_cols if col != TARGET]
    
    # Criar lag de 1 mês APENAS para a variável target
    # Isso evita data leakage: usamos y_{t-1} para prever y_t
    base[f"{TARGET}_lag1"] = base[TARGET].shift(1)
    
    # Features finais = variáveis macro + TARGET_lag1
    feature_cols = predictors + [f"{TARGET}_lag1"]
    
    # Remover linhas com valores ausentes
    final = finalize_dataset(base, feature_cols)
    
    # Estatísticas
    print(f"\n📊 Estatísticas do dataset:")
    print(f"   • Observações finais: {len(final)}")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Primeira data: {final[DATE_COL].min().date()}")
    print(f"   • Última data: {final[DATE_COL].max().date()}")
    print(f"\n📋 Features incluídas:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"   {i}. {feat}")
    
    # Salvar
    output_path = PREP_DIR / "prepared_FULL.parquet"
    final.to_parquet(output_path, index=False)
    
    print(f"\n✅ Dataset preparado salvo: {output_path}")
    
    return final


#%% ==================== EXECUÇÃO PRINCIPAL ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TCC - PREVISÃO DA INADIMPLÊNCIA DE CARTÕES DE CRÉDITO NO BRASIL")
    print("PIPELINE DE DADOS: EXTRAÇÃO E PREPARAÇÃO")
    print("="*80)
    
    # Bloco 1: Extração
    try:
        df_consolidado = extract_all_series()
    except Exception as e:
        print(f"\n❌ ERRO na extração: {e}")
        exit(1)
    
    # Bloco 2: Preparação
    try:
        df_preparado = prepare_full_dataset()
    except Exception as e:
        print(f"\n❌ ERRO na preparação: {e}")
        exit(1)
    
    # Resumo final
    print("\n" + "="*80)
    print("✅ PIPELINE DE DADOS CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\n📁 Arquivos gerados:")
    print(f"   1. {DATA_DIR / 'dados_consolidados_macro_credito.parquet'}")
    print(f"   2. {PREP_DIR / 'prepared_FULL.parquet'}")
    print(f"\n🚀 Próximo passo: Execute o script '02_analysis_pipeline.py'")
    print("="*80 + "\n")
