# Atualização do repositório para a nova arquitetura (2 scripts)

## Arquivos que devem substituir os antigos
Substitua diretamente no repositório:

- `01_data_pipeline.py`
- `02_analysis_pipeline.py`
- `README.md`
- `QUICKSTART.md`
- `requirements.txt`

## Arquivos auxiliares do trabalho atual que deixam de ser necessários no GitHub
Com a consolidação em 2 scripts, estes arquivos podem ser removidos do repositório principal:

- `01_extraction.py`
- `02_preparation.py`
- `02_preparation_COMPLETO.py`
- `03_analysis.py`
- `04_model_diagnostics.py`
- `04_model_diagnostics_chow.py`
- `05_colinearity_analysis.py`
- `06_modelagem_otimizada.py`

## Nova lógica do projeto

### Script 1 — `01_data_pipeline.py`
Consolida:
- extração das séries;
- preparação das bases FULL e EXCL;
- diagnósticos estatísticos da preparação.

### Script 2 — `02_analysis_pipeline.py`
Consolida:
- modelagem final;
- diagnósticos dos modelos;
- Breusch-Pagan;
- Chow test;
- comparações FULL vs EXCL.

## Estrutura de saída esperada
- `data/`
- `prepared/`
- `results_preparation/`
- `results/`

## Ordem de execução
```bash
python 01_data_pipeline.py
python 02_analysis_pipeline.py
```
