# \# Previsão da Inadimplência de Cartões de Crédito no Brasil

# 

# Projeto de Data Science desenvolvido no contexto do MBA em Data Science \& Analytics (USP/ESALQ), com foco na previsão da taxa de inadimplência de cartões de crédito no Brasil a partir de séries macroeconômicas oficiais.

# 

# Este repositório foi estruturado em \*\*2 scripts principais\*\*, concentrando todo o pipeline do trabalho:

# 

# \- `01\_data\_pipeline.py` → extração, preparação e diagnósticos estatísticos

# \- `02\_analysis\_pipeline.py` → modelagem, diagnósticos finais e teste de Chow

# 

# \---

# 

# \## Visão geral do projeto

# 

# O objetivo do estudo é investigar até que ponto variáveis macroeconômicas conseguem explicar e prever a inadimplência total de cartões de crédito no Brasil, além de avaliar como um período de instabilidade estrutural afeta o desempenho preditivo dos modelos.

# 

# A análise compara dois cenários:

# 

# \- \*\*FULL\*\*: série completa disponível

# \- \*\*EXCL\*\*: exclusão do período de 2019 a 2021, para investigar o impacto de uma possível quebra estrutural

# 

# \---

# 

# \## Pergunta central

# 

# \*\*Até que ponto variáveis macroeconômicas conseguem prever a inadimplência total de cartões de crédito no Brasil, e como a presença de um período estruturalmente instável afeta o desempenho dos modelos?\*\*

# 

# \---

# 

# \## Variável alvo

# 

# \- `inadimpl\_cartao\_total`

# 

# \---

# 

# \## Variáveis explicativas utilizadas

# 

# \- `selic\_mensal`

# \- `ibcbr\_dessaz`

# \- `ibcbr\_sem\_ajuste`

# \- `ipca\_mensal`

# \- `comprometimento\_renda`

# \- `endividamento\_familias`

# \- `inadimpl\_cartao\_total\_lag1`

# 

# \---

# 

# \## Fontes de dados

# 

# As séries utilizadas são obtidas a partir de bases oficiais, com foco no Banco Central do Brasil (SGS/BCB), em frequência mensal.

# 

# O pipeline coleta e consolida automaticamente as séries configuradas, gerando uma base única para modelagem.

# 

# \---

# 

# \## Estrutura do pipeline

# 

# \## 1) `01\_data\_pipeline.py`

# 

# Responsável por:

# 

# \- baixar e consolidar as séries do Banco Central do Brasil (SGS);

# \- padronizar a base mensal;

# \- gerar os datasets:

# &#x20; - `prepared/prepared\_FULL.parquet`

# &#x20; - `prepared/prepared\_EXCL.parquet`

# \- executar os diagnósticos estatísticos da etapa de preparação:

# &#x20; - estatísticas descritivas;

# &#x20; - correlação e heatmap;

# &#x20; - scatter-matrix;

# &#x20; - VIF e tolerância;

# &#x20; - testes de normalidade dos resíduos;

# &#x20; - Box-Cox da variável alvo;

# &#x20; - stepwise opcional;

# &#x20; - Shapiro-Francia opcional;

# &#x20; - correlação com `pingouin` opcional.

# 

# \## 2) `02\_analysis\_pipeline.py`

# 

# Responsável por:

# 

# \- carregar os datasets preparados;

# \- rodar a modelagem preditiva final nos cenários FULL e EXCL;

# \- gerar diagnósticos dos modelos;

# \- executar o teste de Breusch-Pagan;

# \- executar o teste de Chow para quebra estrutural;

# \- comparar os cenários FULL e EXCL;

# \- salvar tabelas e gráficos consolidados para uso no TCC.

# 

# \---

# 

# \## Modelos avaliados

# 

# \- Regressão Linear (OLS + Stepwise + Box-Cox)

# \- ARIMA / SARIMAX

# \- Random Forest

# \- XGBoost

# \- MLP

# \- LSTM

# 

# \---

# 

# \## Métricas utilizadas

# 

# \- \*\*MSE\*\*

# \- \*\*R² ajustado\*\*

# \- \*\*R² da variância\*\*

# \- \*\*MAPE\*\*

# \- \*\*Directional Accuracy (DA)\*\*

# 

# \---

# 

# \## Regras metodológicas principais

# 

# \- As variáveis macroeconômicas entram em nível, sem defasagens generalizadas.

# \- É criada apenas a variável `inadimpl\_cartao\_total\_lag1`.

# \- Não há imputação por forward fill.

# \- O cenário \*\*EXCL\*\* remove o intervalo de `2019-01-01` a `2021-12-01`.

# \- Para os modelos lineares:

# &#x20; - o stepwise é executado no conjunto completo;

# &#x20; - depois são removidas as variáveis:

# &#x20;   - `endividamento\_familias`

# &#x20;   - `ibcbr\_sem\_ajuste`

# \- Para ARIMA e demais modelos, essas variáveis também são retiradas conforme a regra metodológica do trabalho.

# 

# \---

# 

# \## Estrutura esperada do projeto

# 

# ```text

# credit-default-prediction-brazil/

# │

# ├── 01\_data\_pipeline.py

# ├── 02\_analysis\_pipeline.py

# ├── README.md

# ├── QUICKSTART.md

# ├── requirements.txt

# ├── LICENSE

# │

# ├── data/

# ├── prepared/

# ├── results\_preparation/

# └── results/

