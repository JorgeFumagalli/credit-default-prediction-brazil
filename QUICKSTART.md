# ⚡ Guia Rápido de Execução

## 🚀 Setup Rápido (5 minutos)

```bash
# 1. Clone e entre no diretório
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale dependências
pip install -r requirements.txt
```

## 📊 Execução Completa

### Opção 1: Tudo de Uma Vez (⏱️ ~45 min)

```bash
# Executa todo o pipeline
python 01_data_pipeline.py && python 02_analysis_pipeline.py
```

### Opção 2: Passo a Passo

```bash
# Passo 1: Extração e Preparação (15-20 min)
python 01_data_pipeline.py

# Passo 2: Análises e Modelagem (30-45 min)
python 02_analysis_pipeline.py
```

## 📁 O Que Será Gerado

Após execução completa, você terá:

```
📂 data/
   ├── dados_consolidados_macro_credito.parquet  (séries do BCB)

📂 prepared/
   └── prepared_FULL.parquet  (dataset pronto)

📂 colinearity_results/
   ├── correlation_FULL.csv
   ├── correlation_EXCL.csv
   ├── vif_FULL.csv
   ├── vif_EXCL.csv
   ├── heatmap_FULL.png
   └── heatmap_EXCL.png

📂 results_diagnostics/
   ├── results_FULL_exploratory.csv
   ├── results_EXCL_exploratory.csv
   ├── linear_coeffs_FULL.csv
   ├── linear_coeffs_EXCL.csv
   ├── xgb_importance_FULL.csv
   └── xgb_importance_EXCL.csv

📂 results_final/
   ├── results_FULL_final.csv
   ├── results_EXCL_final.csv
   ├── results_FULL_EXCL_consolidated.csv
   └── diagnostics/
       ├── linear_coeffs_FULL.csv
       ├── linear_coeffs_EXCL.csv
       ├── xgb_importance_FULL.csv
       └── xgb_importance_EXCL.csv

📂 plots_diagnostics/
   ├── linear_coef_FULL.png
   ├── linear_coef_EXCL.png
   ├── xgb_import_FULL.png
   └── xgb_import_EXCL.png
```

## 🎯 Principais Arquivos de Resultado

### Para Análise de Colinearidade:
- `colinearity_results/vif_FULL.csv` → VIF de todas as variáveis
- `colinearity_results/heatmap_FULL.png` → Matriz de correlação visual

### Para Resultados dos Modelos:
- `results_final/results_FULL_EXCL_consolidated.csv` → Comparação de todos os modelos

### Para Interpretabilidade:
- `results_final/diagnostics/linear_coeffs_FULL.csv` → Impacto de cada variável (Linear)
- `results_final/diagnostics/xgb_importance_FULL.csv` → Importância (XGBoost)

## ⚠️ Troubleshooting

### Erro: "No module named 'tensorflow'"
```bash
pip install tensorflow>=2.15
```

### Erro: "No module named 'xgboost'"
```bash
pip install xgboost>=2.0
```

### Erro: Download das séries falha
- Verifique sua conexão com internet
- O script tenta 2 hosts diferentes automaticamente
- Em caso de falha persistente, os dados podem ser baixados manualmente do SGS/BCB

### Script muito lento
- Normal: extração de dados leva ~15 min
- Modelagem completa leva ~30-45 min
- Processamento inclui treinamento de 10 modelos (5 modelos × 2 cenários)

## 📖 Próximos Passos

1. Leia o [README.md](README.md) completo
2. Explore os resultados em `results_final/`
3. Visualize os gráficos em `plots_diagnostics/`
4. Analise as métricas em `results_FULL_EXCL_consolidated.csv`

## 💡 Dicas

- Use um ambiente com GPU para acelerar treinamento do LSTM/MLP
- Os gráficos são salvos em alta resolução (300 DPI)
- Todos os CSVs podem ser abertos no Excel para análise rápida
- Para reproduzir exatamente os mesmos resultados, as seeds já estão fixadas (42)

## 🎓 Para o TCC

Os principais resultados para o TCC estão em:
- `results_final/results_FULL_final.csv` → Tabela de resultados final
- `plots_diagnostics/` → Gráficos para inclusão no trabalho
- `colinearity_results/heatmap_FULL.png` → Análise de multicolinearidade

---

**Tempo total estimado**: 45-60 minutos
**Espaço em disco**: ~50 MB

🎉 **Boa análise!**
