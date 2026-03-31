# ⚡ Quickstart Guide

Este guia permite executar o pipeline completo do projeto em **2 etapas principais**.

---

## 🚀 1. Setup do ambiente

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

### Instalar dependências
```bash
pip install -r requirements.txt
```

---

## ▶️ 2. Executar o pipeline

### Etapa 1 — extração, preparação e diagnósticos
```bash
python 01_data_pipeline.py
```

### Etapa 2 — modelagem, diagnósticos finais e teste de Chow
```bash
python 02_analysis_pipeline.py
```

---

## 📁 Pastas e arquivos gerados

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

## 🔎 Arquivos principais para conferência

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
- `results/diagnostics/xgb_importance_FULL.csv`

---

## ⏱️ Tempo esperado

O tempo de execução pode variar conforme a máquina, mas em geral:

- `01_data_pipeline.py` → cerca de **5 a 15 minutos**
- `02_analysis_pipeline.py` → cerca de **20 a 45 minutos**

---

## ⚠️ Observações

- O pacote `scipy` é obrigatório.
- Os pacotes `pingouin` e `statstests` **não são obrigatórios**.  
  Se não estiverem instalados, os trechos opcionais de correlação com p-values, Shapiro-Francia e stepwise auxiliar serão apenas ignorados na etapa de preparação.
- A etapa de modelagem utiliza TensorFlow, portanto pode ser mais lenta dependendo do ambiente.

---

## ✅ Ordem correta de execução

Sempre rode nesta ordem:

```bash
python 01_data_pipeline.py
python 02_analysis_pipeline.py
```
