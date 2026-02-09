# ⚡ Quickstart Guide

This guide allows you to run the full project pipeline in approximately **45–60 minutes**.

---

## 🚀 Environment Setup

```bash
# Clone repository
git clone https://github.com/JorgeFumagalli/credit-default-prediction-brazil.git
cd credit-default-prediction-brazil

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux / Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
▶️ Run the Pipeline
Option 1 — Full Execution
python 01_data_pipeline.py && python 02_analysis_pipeline.py
Option 2 — Step by Step
# Step 1: Data extraction and preparation (15–20 min)
python 01_data_pipeline.py

# Step 2: Analysis and modeling (30–45 min)
python 02_analysis_pipeline.py
📁 Outputs Generated
After full execution, the following folders will be created:

data/                     # Raw and consolidated macro data
prepared/                 # Final datasets for modeling
colinearity_results/      # VIF and correlation analysis
results_diagnostics/      # Exploratory modeling results
results_final/            # Optimized final results
plots_diagnostics/        # High-resolution plots
🎯 Key Files to Check
Final comparison of all models:

results_final/results_FULL_EXCL_consolidated.csv

Model interpretability:

results_final/diagnostics/linear_coeffs_*.csv

results_final/diagnostics/xgb_importance_*.csv

Collinearity diagnostics:

colinearity_results/vif_*.csv

colinearity_results/heatmap_*.png

⚠️ Common Issues
Missing TensorFlow
pip install tensorflow>=2.15
Missing XGBoost
pip install xgboost>=2.0
Slow execution
Expected behavior due to time-series processing

Total runtime: ~45–60 minutes

🎓 Academic Usage
Key outputs for academic reporting:

results_final/results_FULL_final.csv

plots_diagnostics/

colinearity_results/heatmap_FULL.png

✅ For full methodology and results interpretation, see README.md.
