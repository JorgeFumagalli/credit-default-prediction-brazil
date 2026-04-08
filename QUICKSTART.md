# Quickstart Guide

Este guia permite executar o pipeline completo do projeto em **2 etapas principais**.

---

## 1. Setup do ambiente

```bash
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil
python -m venv venv
```

### Linux / Mac
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

### Instalar dependências principais
```bash
pip install -r requirements.txt
```

### Instalar dependências opcionais para diagnósticos avançados
```bash
pip install pingouin statstests
```

> O projeto roda sem essas dependências opcionais, mas alguns diagnósticos complementares da etapa de preparação podem ser pulados.

---

## 2. Executar o pipeline

### Etapa 1 — extração, preparação e diagnósticos
```bash
python 01_data_pipeline.py
```

### Etapa 2 — modelagem, diagnósticos finais e teste de Chow
```bash
python 02_analysis_pipeline.py
```

### Execução completa
```bash
python 01_data_pipeline.py && python 02_analysis_pipeline.py
```

---

## Pastas e arquivos gerados

### Após a Etapa 1
- `data/dados_consolidados_macro_credito.parquet`
- `prepared/prepared_FULL.parquet`
- `prepared/prepared_EXCL.parquet`
- `results_preparation/`

### Após a Etapa 2
- `results/results_FULL_final.csv`
- `results/results_EXCL_final.csv`
- `results/results_FULL_EXCL_consolidated.csv`
- `results/chow_test_single_break.csv`
- `results/chow_test_multiple_breaks_2019_2021.csv`
- `results/diagnostics/`

---

## Arquivos principais para conferência

### Preparação
- `prepared/prepared_FULL.parquet`
- `prepared/prepared_EXCL.parquet`

### Resultados finais
- `results/results_FULL_final.csv`
- `results/results_EXCL_final.csv`
- `results/results_FULL_EXCL_consolidated.csv`

### Diagnósticos
- `results/FULL_breusch_pagan.csv`
- `results/EXCL_breusch_pagan.csv`
- `results/chow_test_single_break.csv`
- `results/chow_test_multiple_breaks_2019_2021.csv`
- `results/diagnostics/linear_coeffs_FULL.csv`
- `results/diagnostics/linear_coeffs_EXCL.csv`
- `results/diagnostics/xgb_importance_FULL.csv`
- `results/diagnostics/xgb_importance_EXCL.csv`

---

## Tempo esperado

O tempo de execução pode variar conforme a máquina, mas em geral:

- `01_data_pipeline.py` → cerca de **5 a 15 minutos**
- `02_analysis_pipeline.py` → cerca de **20 a 45 minutos**

---

## Observações

- `scipy` é obrigatório.
- `pingouin` e `statstests` são opcionais.
- A etapa de modelagem utiliza TensorFlow, então o tempo total pode variar bastante conforme o ambiente.

---

## Ordem correta de execução

Sempre rode nesta ordem:

```bash
python 01_data_pipeline.py
python 02_analysis_pipeline.py
```

### Observação sobre a extração de dados

Se já existir uma base consolidada local em `data/dados_consolidados_macro_credito.parquet`, o `01_data_pipeline.py` pode reutilizá-la automaticamente para acelerar reexecuções e permitir testes offline.

Para forçar uma nova extração diretamente do BCB/SGS, execute:

```bash
FORCE_BCB_DOWNLOAD=1 python 01_data_pipeline.py
```
