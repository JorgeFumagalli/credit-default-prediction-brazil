# Credit Card Default Forecasting in Brazil

Projeto de portfólio em Data Science desenvolvido a partir do Trabalho de Conclusão de Curso do MBA em Data Science & Analytics (USP/ESALQ), com foco na previsão da inadimplência agregada de cartões de crédito no Brasil a partir de séries macroeconômicas oficiais.

## Overview

This project investigates a practical forecasting question:

**How can we predict aggregate credit card default in Brazil, and how does macroeconomic instability affect model performance?**

To answer this, the study compares statistical, machine learning and deep learning models under two analytical scenarios:

- **FULL**: complete historical series
- **EXCL**: exclusion of the 2019–2021 period to assess performance outside the most unstable macroeconomic regime

---

## Business Motivation

Aggregate default is a relevant indicator for credit risk, planning, and economic intelligence. Anticipating its trajectory can support:

- credit and provisioning decisions
- risk monitoring in changing macroeconomic environments
- comparison between interpretable and more complex predictive approaches
- better understanding of the role of macroeconomic drivers in aggregate delinquency

---

## Objective

Compare the predictive performance of six models applied to total credit card default in Brazil:

- Linear Regression
- ARIMA/SARIMAX
- Random Forest
- XGBoost
- MLP
- LSTM

---

## Data

- **Source:** Central Bank of Brazil (SGS/BCB)
- **Frequency:** Monthly
- **Period:** Mar/2011 to Jul/2025

### Target variable
- `inadimpl_cartao_total`

### Main explanatory variables
- `selic_mensal`
- `ibcbr_dessaz`
- `ibcbr_sem_ajuste`
- `ipca_mensal`
- `comprometimento_renda`
- `endividamento_familias`
- `inadimpl_cartao_total_lag1`

---

## Methodological Design

The repository is organized around two main scripts:

### `01_data_pipeline.py`
Responsible for:
- extracting official macroeconomic series from BCB/SGS
- validating and consolidating the analytical base
- generating the FULL and EXCL scenarios
- running preparation-stage statistical diagnostics
- exporting intermediate files for reproducibility

### `02_analysis_pipeline.py`
Responsible for:
- training and evaluating the final models in FULL and EXCL
- producing consolidated comparison tables
- generating figures and result summaries
- running statistical diagnostics for the linear specification
- applying the Chow test for structural break analysis

### Key methodological decisions
- time-aware train/test split
- use of only the target `lag1` in the final specification
- stepwise variable selection for the linear model
- Box-Cox transformation in the optimized linear specification
- exclusion of `endividamento_familias` and `ibcbr_sem_ajuste` from the final linear model after diagnostics
- comparison across scenarios to assess sensitivity to structural instability

---

## Main Results

### FULL scenario
| Model | Variance R² | Adjusted R² | MSE | MAPE (%) | DA (%) |
|---|---:|---:|---:|---:|---:|
| ARIMA/SARIMAX | **0.8887** | **0.8695** | **0.0149** | **1.81** | 50.00 |
| Linear | 0.8388 | 0.8287 | 0.0177 | 1.87 | 44.12 |
| XGBoost | 0.7862 | 0.7497 | 0.0263 | 2.26 | 47.06 |
| Random Forest | 0.7831 | 0.7457 | 0.0398 | 3.01 | 44.12 |
| MLP | 0.5936 | 0.5235 | 0.7809 | 12.44 | 32.35 |
| LSTM | 0.5064 | 0.4213 | 0.9465 | 15.53 | 47.06 |

### EXCL scenario
| Model | Variance R² | Adjusted R² | MSE | MAPE (%) | DA (%) |
|---|---:|---:|---:|---:|---:|
| Linear | **0.7793** | **0.7708** | **0.0171** | **1.74** | 37.04 |
| ARIMA/SARIMAX | 0.7593 | 0.7046 | 0.0230 | 1.98 | 51.85 |
| Random Forest | 0.6621 | 0.5853 | 0.0390 | 2.98 | 33.33 |
| MLP | 0.6150 | 0.5275 | 0.4877 | 10.89 | 44.44 |
| LSTM | 0.6067 | 0.5173 | 0.4329 | 9.02 | **62.96** |
| XGBoost | 0.5559 | 0.4550 | 0.0302 | 2.50 | 44.44 |

### Executive interpretation
- In the **FULL** scenario, **ARIMA/SARIMAX** delivered the strongest overall performance.
- In the **EXCL** scenario, **Linear Regression** became the leading model.
- The results suggest that temporal structure and macroeconomic regime were central to predictive performance.
- More complex architectures did not consistently outperform statistical benchmarks.

---

## Structural Break Evidence

The project also tested whether the pandemic period represented a relevant break in the series dynamics.

### Chow test highlights
- **Jan/2019:** not significant
- **Jan/2020:** statistically significant
- **Jan/2021:** statistically significant

This supports the decision to compare FULL and EXCL scenarios rather than relying on a single modeling environment.

---

## Key Insights

1. **Model performance depends on the economic regime being analyzed.**  
   The leading model changed when the unstable period was removed.

2. **Temporal structure mattered more than model complexity in this application.**  
   Statistical approaches remained highly competitive.

3. **Interpretability remains valuable.**  
   The linear model combined strong performance with clearer economic reading.

4. **Methodological rigor improves portfolio quality.**  
   Diagnostics, structural break testing and scenario comparison strengthen the analytical narrative.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

---

## Repository Structure

```text
credit-default-prediction-brazil/
├── 01_data_pipeline.py
├── 02_analysis_pipeline.py
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── data/
├── prepared/
├── results_preparation/
└── results/
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil
```

### 2. Create and activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python 01_data_pipeline.py
python 02_analysis_pipeline.py
```

---

## Outputs

The project generates:
- prepared datasets for FULL and EXCL
- consolidated comparison tables
- actual vs. predicted plots by model
- metric comparison figures
- statistical diagnostic outputs
- structural break test results

---

## Why this project matters in my portfolio

This project demonstrates:
- reproducible analytical pipeline design
- integration between statistical modeling and machine learning
- critical comparison between simple and complex models
- business-oriented interpretation of technical results
- application to risk, credit and forecasting problems

---

## Limitations and Next Steps

### Limitations
- monthly series with a moderate number of observations
- aggregate default instead of transaction-level behavior
- limited macroeconomic feature set

### Next steps
- expand the exogenous variable set
- test hybrid and Bayesian approaches
- evaluate rolling-window stability
- explore forecast intervals and regime-aware calibration

---

## Author

**Jorge Luiz Fumagalli**

- LinkedIn: https://www.linkedin.com/in/jorge-fumagalli/
- GitHub: https://github.com/JorgeFumagalli

