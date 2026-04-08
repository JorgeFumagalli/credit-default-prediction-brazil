# Previsão da Inadimplência de Cartões de Crédito no Brasil

Projeto de Data Science desenvolvido no contexto do **MBA em Data Science & Analytics (USP/ESALQ)**, com foco na **previsão da taxa de inadimplência de cartões de crédito no Brasil** a partir de séries macroeconômicas oficiais.

O repositório foi reorganizado em **2 scripts principais**, concentrando o pipeline completo do trabalho:

- `01_data_pipeline.py` → extração, preparação e diagnósticos estatísticos
- `02_analysis_pipeline.py` → modelagem, diagnósticos dos modelos e teste de Chow

---

## Objetivo

Avaliar o desempenho de modelos estatísticos, de machine learning e de deep learning na previsão da inadimplência de cartões, comparando:

- um cenário **FULL**, com toda a série disponível;
- um cenário **EXCL**, excluindo o período de **2019 a 2021** para investigar possíveis efeitos de instabilidade estrutural.

---

## Pergunta central

**Até que ponto variáveis macroeconômicas conseguem explicar e prever a inadimplência total de cartões de crédito no Brasil, e como a presença de um período estruturalmente instável afeta o desempenho dos modelos?**

---

## Variável alvo

- `inadimpl_cartao_total`

## Variáveis explicativas utilizadas

- `selic_mensal`
- `ibcbr_dessaz`
- `ibcbr_sem_ajuste`
- `ipca_mensal`
- `comprometimento_renda`
- `endividamento_familias`
- `inadimpl_cartao_total_lag1`

---

## Fontes de dados

As séries são obtidas a partir de bases oficiais, com foco no **Banco Central do Brasil (SGS/BCB)**, em frequência mensal.

O pipeline coleta e consolida automaticamente as séries configuradas, gerando a base final utilizada na modelagem.

---

## Estrutura do pipeline

### 1) `01_data_pipeline.py`

Responsável por:

- baixar e consolidar as séries do Banco Central do Brasil (SGS);
- padronizar a base mensal;
- gerar os datasets:
  - `prepared/prepared_FULL.parquet`
  - `prepared/prepared_EXCL.parquet`
- executar os diagnósticos estatísticos da etapa de preparação:
  - estatísticas descritivas;
  - correlação e heatmap;
  - scatter-matrix;
  - VIF e tolerância;
  - testes de normalidade dos resíduos;
  - Box-Cox da variável alvo;
  - stepwise opcional;
  - Shapiro-Francia opcional;
  - correlação com `pingouin` opcional.

### 2) `02_analysis_pipeline.py`

Responsável por:

- carregar os datasets preparados;
- rodar a modelagem preditiva final nos cenários FULL e EXCL;
- gerar diagnósticos dos modelos;
- executar o teste de Breusch-Pagan;
- executar o teste de Chow para quebra estrutural;
- comparar os cenários FULL e EXCL;
- salvar tabelas e gráficos consolidados para uso no TCC.

---

## Modelos avaliados

- Regressão Linear (OLS + Stepwise + Box-Cox)
- ARIMA / SARIMAX
- Random Forest
- XGBoost
- MLP
- LSTM

---

## Métricas utilizadas

- **MSE**
- **R² ajustado**
- **R² da variância**
- **MAPE**
- **Directional Accuracy (DA)**

---

## Regras metodológicas principais

- As variáveis macroeconômicas entram em nível, sem defasagens generalizadas.
- É criada apenas a variável `inadimpl_cartao_total_lag1`.
- Não há imputação por forward fill.
- O cenário **EXCL** remove o intervalo de `2019-01-01` a `2021-12-01`.
- Para os modelos lineares:
  - o stepwise é executado no conjunto completo;
  - depois são removidas as variáveis:
    - `endividamento_familias`
    - `ibcbr_sem_ajuste`
- Para ARIMA e demais modelos, essas variáveis também são retiradas conforme a regra metodológica do trabalho.

---

## Estrutura esperada do projeto

```text
credit-default-prediction-brazil/
│
├── 01_data_pipeline.py
├── 02_analysis_pipeline.py
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
├── prepared/
├── results_preparation/
└── results/
```

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil
```

Crie e ative um ambiente virtual:

```bash
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

Instale as dependências principais:

```bash
pip install -r requirements.txt
```

---

## Dependências opcionais para diagnósticos avançados

Algumas saídas complementares do `01_data_pipeline.py` dependem de bibliotecas opcionais:

```bash
pip install pingouin statstests
```

### O que acontece se elas não estiverem instaladas?

O pipeline principal continua funcionando, mas algumas análises adicionais podem ser puladas, como:

- `rcorr` com `pingouin`;
- teste de **Shapiro-Francia**;
- stepwise auxiliar baseado no pacote `statstests`.

Isso não impede a execução do projeto, mas reduz a completude dos diagnósticos estatísticos da etapa de preparação.

---


### Observação sobre a extração de dados

Se já existir uma base consolidada local em `data/dados_consolidados_macro_credito.parquet`, o `01_data_pipeline.py` pode reutilizá-la automaticamente para acelerar reexecuções e permitir testes offline.

Para forçar uma nova extração diretamente do BCB/SGS, execute:

```bash
FORCE_BCB_DOWNLOAD=1 python 01_data_pipeline.py
```

## Como executar

### Execução completa

```bash
python 01_data_pipeline.py && python 02_analysis_pipeline.py
```

### Execução em etapas

#### Etapa 1 — extração, preparação e diagnósticos
```bash
python 01_data_pipeline.py
```

#### Etapa 2 — modelagem, comparações e teste de Chow
```bash
python 02_analysis_pipeline.py
```

---

## Principais saídas geradas

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
- `results/FULL_best_real_vs_pred.png`
- `results/EXCL_best_real_vs_pred.png`
- `results/FULL_vs_EXCL_inadimplencia.png`
- `results/PANEL_FULL_EXCL_metrics.png`
- `results/chow_test_single_break.csv`
- `results/chow_test_multiple_breaks_2019_2021.csv`
- `results/diagnostics/*`

---

## Destaques do projeto

Este projeto combina:

- modelagem estatística;
- machine learning;
- deep learning;
- diagnóstico de colinearidade;
- testes de normalidade dos resíduos;
- teste de heterocedasticidade;
- teste de quebra estrutural;
- comparação entre regimes econômicos distintos.

Além da previsão em si, o trabalho busca equilibrar **desempenho preditivo**, **interpretação econômica** e **coerência metodológica**.

---

## Aplicação acadêmica

O repositório foi desenvolvido como base analítica para o TCC do MBA em Data Science & Analytics, servindo tanto para:

- reprodução do pipeline completo;
- geração de tabelas e gráficos para o trabalho;
- documentação da metodologia utilizada.

---

## Autor

**Jorge Luiz Fumagalli**

- LinkedIn: https://www.linkedin.com/in/jorge-fumagalli/
- GitHub: https://github.com/JorgeFumagalli

---

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
