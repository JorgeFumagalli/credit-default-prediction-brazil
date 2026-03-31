# 📈 Previsão da Inadimplência de Cartões de Crédito no Brasil

Projeto de Data Science desenvolvido no contexto do **MBA em Data Science & Analytics (USP/ESALQ)**, com foco em **prever a taxa de inadimplência de cartões de crédito no Brasil** a partir de séries macroeconômicas oficiais.

A estrutura do repositório foi organizada em **apenas 2 scripts principais**, concentrando todo o pipeline do trabalho:

- `01_data_pipeline.py` → extração, preparação e diagnósticos estatísticos
- `02_analysis_pipeline.py` → modelagem, diagnósticos finais e teste de Chow

---

## 🎯 Objetivo

Avaliar o desempenho de modelos estatísticos, de machine learning e de deep learning na previsão da inadimplência de cartões, comparando:

- um cenário **FULL** com toda a série disponível;
- um cenário **EXCL**, excluindo o período de **2019 a 2021** para investigar os efeitos de instabilidade estrutural.

---

## 🧠 Pergunta central

Até que ponto variáveis macroeconômicas conseguem explicar e prever a inadimplência total de cartões de crédito no Brasil, e como a presença de um período estruturalmente instável afeta o desempenho dos modelos?

---

## 📊 Variável alvo

- `inadimpl_cartao_total`

## 📌 Variáveis explicativas utilizadas

- `selic_mensal`
- `ibcbr_dessaz`
- `ibcbr_sem_ajuste`
- `ipca_mensal`
- `comprometimento_renda`
- `endividamento_familias`
- `inadimpl_cartao_total_lag1`

---

## 🗂️ Estrutura do pipeline

### 1) `01_data_pipeline.py`
Responsável por:

- baixar e consolidar as séries do **Banco Central do Brasil (SGS)**;
- padronizar a base mensal;
- gerar os datasets:
  - `prepared/prepared_FULL.parquet`
  - `prepared/prepared_EXCL.parquet`
- executar os diagnósticos estatísticos da preparação:
  - estatísticas descritivas;
  - correlação e heatmap;
  - scatter-matrix;
  - VIF e tolerância;
  - testes de normalidade dos resíduos;
  - Box-Cox da variável alvo;
  - stepwise opcional, se o pacote estiver disponível.

### 2) `02_analysis_pipeline.py`
Responsável por:

- carregar os datasets preparados;
- rodar a modelagem preditiva final;
- gerar diagnósticos dos modelos;
- executar o **teste de Breusch-Pagan**;
- executar o **teste de Chow**;
- comparar os cenários FULL e EXCL.

---

## 🤖 Modelos avaliados

- Regressão Linear (OLS + Stepwise + Box-Cox)
- ARIMA / SARIMAX
- Random Forest
- XGBoost
- MLP
- LSTM

---

## 📏 Métricas utilizadas

- **MSE**
- **R² ajustado**
- **R² da variância**
- **MAPE**
- **Directional Accuracy (DA)**

---

## 🧪 Regras metodológicas principais

- As variáveis macroeconômicas entram em **nível**, sem defasagens generalizadas.
- É criada apenas a variável `inadimpl_cartao_total_lag1`.
- O cenário **EXCL** remove o intervalo de **2019-01-01 a 2021-12-01**.
- Para os modelos lineares:
  - o stepwise é executado no conjunto completo;
  - depois são removidas as variáveis:
    - `endividamento_familias`
    - `ibcbr_sem_ajuste`
- Para ARIMA e demais modelos, essas variáveis também são retiradas conforme a regra metodológica do trabalho.

---

## 📁 Estrutura esperada do projeto

```text
credit-default-prediction-brazil/
│
├── 01_data_pipeline.py
├── 02_analysis_pipeline.py
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── LICENSE
│
├── data/
├── prepared/
├── results_preparation/
└── results/
```

---

## ▶️ Como executar

Consulte o arquivo:

**`QUICKSTART.md`**

---

## 📤 Principais saídas geradas

### Etapa 1 — preparação
- `data/dados_consolidados_macro_credito.parquet`
- `prepared/prepared_FULL.parquet`
- `prepared/prepared_EXCL.parquet`
- `results_preparation/*`

### Etapa 2 — análise
- `results/results_FULL_final.csv`
- `results/results_EXCL_final.csv`
- `results/results_FULL_EXCL_consolidated.csv`
- `results/FULL_*_real_vs_pred.png`
- `results/EXCL_*_real_vs_pred.png`
- `results/chow_test_single_break.csv`
- `results/chow_test_multiple_breaks_2019_2021.csv`
- `results/diagnostics/*`

---

## 🛠️ Tech stack

- Python
- Pandas / NumPy
- SciPy
- Statsmodels
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib
- Requests

---

## 👤 Autor

**Jorge Luiz Fumagalli**

- LinkedIn: `https://www.linkedin.com/in/jorge-fumagalli/`
- GitHub: `https://github.com/JorgeFumagalli`

---

## 📄 Licença

Este projeto utiliza a licença **MIT**.