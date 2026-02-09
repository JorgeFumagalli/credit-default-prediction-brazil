# 📈 Credit Card Default Prediction in Brazil

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic](https://img.shields.io/badge/Academic%20Project-USP%2FESALQ-red.svg)](https://esalq.usp.br/)

This repository contains an end-to-end Data Science project focused on predicting credit card default rates in Brazil using macroeconomic time-series data.

The project was developed as part of the **MBA in Data Science & Analytics (USP/ESALQ)** and combines academic rigor with real-world credit risk modeling practices.

---

## 📊 Project Overview

The objective of this project is to analyze and forecast the **total credit card default rate** in Brazil using monthly macroeconomic indicators published by official institutions such as the **Banco Central do Brasil (BCB)** and **IBGE**.

An automated pipeline was built to handle:
- Data extraction and consolidation
- Feature engineering and preprocessing
- Collinearity diagnostics
- Machine Learning and Deep Learning model training
- Scenario-based performance comparison

---

## 🎯 Objectives

- Compare supervised learning models for credit default forecasting
- Evaluate model robustness under different economic regimes
- Assess the impact of structural shocks (pandemic period)
- Provide practical guidance for credit risk model selection
- Balance predictive performance and interpretability

---

## 💡 Key Insights

### 1. Economic context matters more than model complexity
- **LSTM** achieved superior performance in periods of high volatility
- **SVR** outperformed deep learning models in stable economic environments

### 2. Deep learning requires sufficient temporal depth
- MLP models showed severe overfitting due to limited time-series length
- LSTM remained competitive due to its sequence-aware architecture

### 3. Linear models remain strong baselines
- Linear Regression explained over 60% of default variability in the full sample
- Simple models remain valuable for interpretability and governance

---

## 🚀 Main Results

### Scenario FULL (2015–2025 | Includes Pandemic)

| Model | MSE | R² | MAPE (%) | DA (%) | Notes |
|------|-----|-----|----------|--------|-------|
| **LSTM** ⭐ | **0.0179** | **0.7050** | **1.83** | 40.00 | Best under high volatility |
| Linear Regression | 0.0210 | 0.6542 | 2.05 | 44.00 | Strong baseline |
| XGBoost | 0.0228 | 0.6242 | 2.13 | 44.00 | Balanced performance |
| SVR | 0.0572 | 0.0594 | 3.10 | 56.00 | Highest directional accuracy |
| MLP | 14.9447 | -244.79 | 56.59 | 48.00 | Severe overfitting |

---

### Scenario EXCL (Excluding 2019–2021)

| Model | MSE | R² | MAPE (%) | DA (%) | Notes |
|------|-----|-----|----------|--------|-------|
| **SVR** ⭐ | **0.0295** | **0.3559** | **2.26** | 35.29 | Best in stable regime |
| Linear Regression | 0.0370 | 0.1924 | 2.57 | 47.06 | Consistent |
| XGBoost | 0.1422 | -2.1029 | 5.40 | 41.18 | Loss of generalization |
| LSTM | 0.2194 | -3.7858 | 7.50 | 47.06 | Data-hungry |
| MLP | 0.9264 | -19.2102 | 12.36 | 41.18 | Inadequate |

---

## 📊 Data Description

### Data Sources
- **Banco Central do Brasil (BCB)** – SGS
- **IBGE** – IPCA
- Monthly data from **Jan/2015 to Jul/2025**

### Target Variable
- Total credit card default rate (%)

### Predictors
- Selic interest rate
- IBC-Br (economic activity proxy)
- Inflation (IPCA)
- Household income commitment

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Statsmodels
- Matplotlib, Seaborn

---

## 📁 Project Structure

credit-default-prediction-brazil/
│
├── 01_data_pipeline.py
├── 02_analysis_pipeline.py
│
├── data/
├── prepared/
├── colinearity_results/
├── results_diagnostics/
├── results_final/
├── plots_diagnostics/
│
├── requirements.txt
├── QUICKSTART.md
├── README.md
└── LICENSE


---

## ▶️ How to Run

For a fast setup and execution guide, see:

👉 **[QUICKSTART.md](QUICKSTART.md)**

---

## 👤 Author

**Jorge Luiz Fumagalli**

🎓 MBA in Data Science & Analytics – USP/ESALQ  
🎓 BSc in Production Engineering  

🔗 LinkedIn: https://www.linkedin.com/in/jorge-fumagalli-bb8975121/  
🐙 GitHub: https://github.com/JorgeFumagalli  

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this project in academic or professional work, please cite:

```bibtex
@mastersthesis{fumagalli2026,
  author  = {Fumagalli, Jorge Luiz},
  title   = {Credit Card Default Prediction in Brazil Using Machine Learning},
  school  = {USP/ESALQ},
  year    = {2026}
}
⭐ If you find this project useful, consider giving it a star!