# 📈 Previsão da Inadimplência de Cartões de Crédito no Brasil

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TCC](https://img.shields.io/badge/TCC-USP%2FESALQ-red.svg)](https://esalq.usp.br/)

> Trabalho de Conclusão de Curso (TCC) do MBA em Data Science & Analytics (USP/ESALQ): Análise comparativa de modelos de Machine Learning e Deep Learning para previsão de inadimplência, avaliando performance em diferentes regimes econômicos.

---

## 📊 Visão Geral

Este projeto analisa a previsão de inadimplência total de cartões de crédito no Brasil utilizando variáveis macroeconômicas mensais entre janeiro de 2015 e julho de 2025.

### 🎯 Objetivos do Projeto

- Comparar performance de 5 modelos supervisionados: **Linear Regression, SVR, XGBoost, MLP e LSTM**
- Avaliar impacto de choques estruturais (pandemia 2019-2021) no desempenho dos modelos
- Identificar qual arquitetura é mais adequada para diferentes regimes econômicos
- Fornecer subsídios práticos para seleção de técnicas em gestão de risco de crédito

### 🏆 Principais Contribuições

1. **Análise Dual de Cenários**: Comparação entre série completa (FULL) vs período estável (EXCL)
2. **Descoberta Metodológica**: LSTM superior em alta volatilidade, SVR em estabilidade
3. **Aplicação Prática**: Orientação para seleção de modelos conforme contexto econômico
4. **Rigor Acadêmico**: Metodologia completa com validação temporal e múltiplas métricas

---

## 🚀 Principais Resultados

### ✅ Cenário FULL (Série Completa 2015-2025)

Inclui período de instabilidade fiscal 2019-2021.

| Modelo | MSE | R² | MAPE (%) | DA (%) | Destaque |
|--------|-----|-----|----------|---------|----------|
| **LSTM** ⭐ | **0.0179** | **0.7050** | **1.83** | 40.00 | Melhor para alta volatilidade |
| Linear Regression | 0.0210 | 0.6542 | 2.05 | 44.00 | Baseline competitivo |
| XGBoost | 0.0228 | 0.6242 | 2.13 | 44.00 | Bom equilíbrio |
| SVR | 0.0572 | 0.0594 | 3.10 | 56.00 | Maior acerto direcional |
| MLP | 14.9447 | -244.79 | 56.59 | 48.00 | Overfitting severo |

> **💡 Insight Chave:** LSTM captura dependências temporais complexas em ambientes de alta volatilidade, explicando 70% da variância da inadimplência.

### ✅ Cenário EXCL (Excluindo 2019-2021)

Remove período de instabilidade para analisar performance em ambiente estável.

| Modelo | MSE | R² | MAPE (%) | DA (%) | Destaque |
|--------|-----|-----|----------|---------|----------|
| **SVR** ⭐ | **0.0295** | **0.3559** | **2.26** | 35.29 | Melhor para estabilidade |
| Linear Regression | 0.0370 | 0.1924 | 2.57 | 47.06 | Consistente |
| XGBoost | 0.1422 | -2.1029 | 5.40 | 41.18 | Perde generalização |
| LSTM | 0.2194 | -3.7858 | 7.50 | 47.06 | Requer mais dados |
| MLP | 0.9264 | -19.2102 | 12.36 | 41.18 | Inadequado |

> **💡 Descoberta:** SVR supera LSTM em ambiente estável, revelando que padrões não-lineares suaves são melhor capturados por kernels RBF sem necessidade de memória temporal complexa.

---

## 💡 Principais Descobertas

### 🎯 Descoberta 1: Contexto Econômico > Complexidade do Modelo

**No cenário FULL (alta volatilidade):**
- **LSTM:** R² = 0.70, MAPE = 1.83%
- Capacidade de capturar dependências temporais durante choques macroeconômicos
- Volatilidade extrema da pandemia exige memória de longo prazo

**No cenário EXCL (estabilidade):**
- **SVR:** R² = 0.36, MAPE = 2.26%
- Padrões não-lineares mais suaves favorecem kernel RBF
- Modelos mais simples suficientes sem choques estruturais

**Implicação Prática:** A escolha do modelo deve considerar o regime econômico vigente, não apenas métricas de treino.

### 🎯 Descoberta 2: Trade-off entre Complexidade e Volume de Dados

- **MLP:** Performance ruim em ambos cenários
- Séries temporais curtas (126 meses) insuficientes para deep learning complexo
- LSTM funciona por ter arquitetura especializada em sequências
- **Lição:** Deep learning requer > 200-300 observações para generalizar bem

### 🎯 Descoberta 3: Baseline Linear Surpreendentemente Competitivo

- **Linear Regression:** R² = 0.65 (FULL), 0.19 (EXCL)
- 65% da inadimplência explicada por relações aproximadamente lineares
- Modelos simples podem ser suficientes para interpretabilidade
- **Lição:** Sempre compare com baseline antes de usar modelos complexos

---

## 📊 Dados e Variáveis

### Fonte dos Dados

- **Banco Central do Brasil** - Sistema Gerenciador de Séries Temporais (SGS)
- **IBGE** - Índice Nacional de Preços ao Consumidor Amplo (IPCA)
- **Período:** Janeiro/2015 a Julho/2025 (126 observações mensais)

### Variáveis Preditoras

| Variável | Descrição | Fonte |
|----------|-----------|-------|
| **Taxa Selic** | Taxa básica de juros da economia brasileira | BCB |
| **IBC-Br Dessazonalizado** | Índice de Atividade Econômica (proxy do PIB) | BCB |
| **IPCA** | Inflação mensal oficial | IBGE |
| **Comprometimento de Renda** | % da renda comprometida com dívidas | BCB |

### Variável Target

- **Inadimplência Total de Cartão de Crédito** (% do saldo total inadimplente)
- Fonte: Banco Central do Brasil
- Série oficial mensal

---

## 🛠️ Tecnologias Utilizadas

### Core Libraries

```
pandas>=2.0          # Manipulação de dados
numpy>=1.24          # Computação numérica
scikit-learn>=1.3    # Machine Learning tradicional
xgboost>=2.0         # Gradient Boosting
tensorflow>=2.15     # Deep Learning
```

### Analysis & Visualization

```
matplotlib>=3.7      # Visualizações
seaborn>=0.13        # Gráficos estatísticos
statsmodels>=0.14    # Análise estatística
```

---

## 📁 Estrutura do Projeto

```
credit-default-prediction-brazil/
│
├── 01_data_pipeline.py          # Script 1: Extração e Preparação
├── 02_analysis_pipeline.py      # Script 2: Análises e Modelagem
│
├── data/
│   ├── raw/                     # Dados brutos do BCB
│   └── processed/               # Dados processados
│
├── prepared/                    # Datasets prontos para modelagem
│
├── colinearity_results/         # Análise de colinearidade
│   ├── correlation_FULL.csv
│   ├── vif_FULL.csv
│   └── heatmap_FULL.png
│
├── results_diagnostics/         # Modelagem exploratória
│   ├── results_FULL_exploratory.csv
│   ├── linear_coeffs_FULL.csv
│   └── xgb_importance_FULL.csv
│
├── results_final/               # Modelagem otimizada
│   ├── results_FULL_final.csv
│   ├── results_EXCL_final.csv
│   └── diagnostics/
│
├── plots_diagnostics/           # Visualizações
│
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
└── LICENSE                      # Licença MIT
```

---

## 🎯 Como Usar

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Executar Pipeline de Dados

```bash
# Script 1: Extração e Preparação (15-20 min)
python 01_data_pipeline.py

# Isso irá:
# - Baixar séries do BCB/SGS
# - Consolidar dados
# - Criar features
# - Preparar dataset para modelagem
```

**Saídas esperadas:**
- `data/dados_consolidados_macro_credito.parquet`
- `prepared/prepared_FULL.parquet`

### 3. Executar Análises e Modelagem

```bash
# Script 2: Análises Completas (30-45 min)
python 02_analysis_pipeline.py

# Isso irá:
# BLOCO 1: Análise de Colinearidade (VIF + Correlação)
# BLOCO 2: Modelagem Exploratória (5 modelos)
# BLOCO 3: Modelagem Otimizada (features selecionadas)
```

**Saídas esperadas:**
- Matrizes de correlação e VIF
- Resultados de todos os modelos (FULL e EXCL)
- Coeficientes e importâncias de variáveis
- Gráficos Real vs Predito
- Diagnósticos completos

---

## 📊 Interpretando os Resultados

### Métricas Utilizadas

- **MSE (Mean Squared Error)**: Erro quadrático médio (quanto menor, melhor)
- **R² (R-squared)**: Proporção da variância explicada (0-1, quanto maior, melhor)
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual absoluto médio
- **DA (Directional Accuracy)**: % de acertos na direção da variação

### Arquivos de Resultados

#### Colinearidade
- `correlation_FULL.csv`: Matriz de correlação completa
- `vif_FULL.csv`: Variance Inflation Factor por variável
- `heatmap_FULL.png`: Visualização da correlação

#### Modelagem Exploratória
- `results_FULL_exploratory.csv`: Métricas de todos os modelos
- `linear_coeffs_FULL.csv`: Coeficientes padronizados da regressão
- `xgb_importance_FULL.csv`: Importância das variáveis no XGBoost

#### Modelagem Otimizada
- `results_FULL_final.csv`: Resultados finais (features selecionadas)
- `results_FULL_EXCL_consolidated.csv`: Comparação entre cenários

---

## 🔮 Trabalhos Futuros

### Melhorias Planejadas
- [ ] Incorporar variáveis microeconômicas (renda per capita, desemprego)
- [ ] Testar modelos híbridos (ensemble ML + DL)
- [ ] Implementar detecção automática de quebras estruturais
- [ ] Sistema de seleção automática de modelo baseado em volatilidade
- [ ] Previsão probabilística (intervalos de confiança)

### Extensões Acadêmicas
- [ ] Análise de outras modalidades de crédito (consignado, veículos)
- [ ] Comparação internacional (Brasil vs outros emergentes)
- [ ] Análise de causalidade (Granger, VAR)
- [ ] Incorporar variáveis de política monetária

---

## 👤 Autor

**Jorge Luiz Fumagalli**

**Formação:**
- 🎓 MBA em Data Science & Analytics - USP/ESALQ (2024-2026)
- 🎓 Engenharia de Produção - UFTM
- 🎓 Técnico em Informática - ETEC

**Orientador do TCC:**
- Prof. Me. Diego Pedroso dos Santos

**Contato:**
- 💼 LinkedIn: [linkedin.com/in/jorge-fumagalli](https://www.linkedin.com/in/jorge-fumagalli-bb8975121/)
- 📧 Email: jorgefumagalli@yahoo.com.br
- 🐙 GitHub: [github.com/JorgeFumagalli](https://github.com/JorgeFumagalli)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🙏 Agradecimentos

- Prof. Diego Pedroso dos Santos pela orientação
- USP/ESALQ pelo programa de MBA em Data Science & Analytics
- Banco Central do Brasil pela disponibilização dos dados
- Comunidades open-source de Machine Learning e Deep Learning

---

## 📖 Citação

Se este trabalho foi útil para sua pesquisa, considere citar:

```bibtex
@mastersthesis{fumagalli2026,
  author  = {Fumagalli, Jorge Luiz},
  title   = {Previsão da Inadimplência de Cartões de Crédito no Brasil com Modelos de Aprendizado de Máquina},
  school  = {USP/ESALQ - MBA em Data Science & Analytics},
  year    = {2026},
  type    = {Trabalho de Conclusão de Curso}
}
```

---

## ⭐ Se este projeto foi útil, considere dar uma estrela!

---

**💡 Dúvidas? Sugestões? Feedbacks são sempre bem-vindos!**

[Abrir Issue](https://github.com/JorgeFumagalli/credit-default-prediction-brazil/issues) | [Pull Requests](https://github.com/JorgeFumagalli/credit-default-prediction-brazil/pulls)
