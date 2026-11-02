# SpendSmart AI: AI-Powered Personal Finance & Spending Pattern Analyzer

![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)
![Tests](https://img.shields.io/badge/Pytest-20%2F20%20Passed-brightgreen.svg)
![License](https://img.shields.io/badge/License-Academic%20Open-green.svg)

**SpendSmart AI** is a production-grade, end-to-end Machine Learning personal finance analytics application designed for a third-year Artificial Intelligence and Data Science engineering curriculum. It demonstrates all **11 Machine Learning Practicals** while solving realistic personal finance challenges: cash flow tracking, expense forecasting, customer spending segmentation, spatial anomaly detection, and reinforcement learning budget simulation.

---

## 🌟 Key Features

1. **Interactive Financial Analytics:** Complete cash flow tracking, income vs. expense breakdowns, category share-of-wallet donut charts, and category budget variance diagnostics.
2. **Predictive Expense Forecasting:** Multivariate Linear Regression, Polynomial Regression (degrees 1, 2, 3), Ridge ($L_2$), Lasso ($L_1$), Random Forest, and Gradient Boosting models evaluated via **chronological splitting** (zero data leakage).
3. **Customer Spending Segmentation:** Standardized K-Means clustering with Elbow curve and Silhouette score validation, deriving interpretable financial personas (e.g., Frugal Savers, Dining Enthusiasts, Tech Spenders).
4. **Spatial Density Anomaly Detection:** DBSCAN clustering that detects arbitrary cluster shapes and isolates unusual transactions as Noise ($-1$) with automatic Z-score diagnostic reasons.
5. **Structural Dimensionality Reduction:** Direct comparison of unsupervised variance maximization (**PCA**) vs. supervised class separability (**LDA**).
6. **Adaptive Reinforcement Learning Budget Simulator:** NumPy-based tabular **Q-Learning** agent operating over a $27$-state discrete Markov Decision Process (MDP) to dynamically optimize budget allocations while **strictly protecting essential living expenses**.
7. **Comprehensive ML Laboratory:** A dedicated interactive laboratory providing mathematical descriptions, parameter controls, live model execution, residual plots, and exportable predictions for all 11 practicals.

---

## 📚 Machine Learning Practicals Matrix

| Practical # | Topic / Algorithm | Module Location | Dashboard Page | Key Concepts & Outputs |
|---|---|---|---|---|
| **Practical 1** | Preprocessing & Feature Engineering | `src/preprocessing.py`, `src/feature_engineering.py` | Page 1, Page 5 | Leakage-free `ColumnTransformer`, group-median imputation, `StandardScaler`, `OneHotEncoder`, rolling $7$d/$30$d stats. |
| **Practical 2** | PCA & LDA Dimensionality Reduction | `src/dimensionality_reduction.py` | Page 3, Page 5 | Unsupervised variance (PCA) vs. Supervised scatter ratio (LDA), 2D projections, classification accuracy before & after. |
| **Practical 3** | Multiple Linear Regression | `src/regression_models.py` | Page 2, Page 5 | Closed-form OLS, chronological split, MSE, RMSE, $R^2$, actual vs. predicted, residual homoscedasticity plot. |
| **Practical 4** | Polynomial Regression | `src/regression_models.py` | Page 2, Page 5 | Degrees 1, 2, 3 comparison, underfitting vs. overfitting curves, bias-variance trade-off analysis. |
| **Practical 5** | Ridge & Lasso Regularization | `src/regression_models.py` | Page 2, Page 5 | $L_2$ weight shrinkage (Ridge) vs. $L_1$ feature selection/sparsity (Lasso), correlation multicollinearity matrix. |
| **Practical 6** | Logistic Regression | `src/classification_models.py` | Page 5 | Binary classification of high-intensity spending without target leakage, confusion matrix, precision, recall, F1, ROC-AUC. |
| **Practical 7** | Support Vector Machines (SVM) | `src/classification_models.py` | Page 5 | Linear kernel vs. Non-linear RBF kernel, margin maximization, support vector count, execution time benchmark. |
| **Practical 8** | K-Means Clustering | `src/clustering_models.py` | Page 3, Page 5 | StandardScaler, Elbow curve (Inertia), Silhouette scores, 2D PCA projection, spending persona profiles. |
| **Practical 9** | DBSCAN Anomaly Detection | `src/anomaly_detection.py` | Page 3, Page 5 | Non-spherical density clustering, noise point isolation ($-1$), noise percentage, category Z-score explanations. |
| **Practical 10** | Ensemble Learning | `src/regression_models.py`, `src/classification_models.py` | Page 2, Page 5 | Bagging (Random Forest) vs. Boosting (HistGradientBoosting) vs. Baselines across regression & classification tasks. |
| **Practical 11** | Reinforcement Learning (Q-Learning) | `src/reinforcement_learning.py` | Page 4, Page 5 | Tabular Bellman optimality updates, 27 states, 4 actions, $\varepsilon$-greedy exploration decay, 12-month simulation. |

---

## 🏗️ Folder Structure

```
spendsmart-ai/
├── app.py                         # Main landing application and dataset manager
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive project guide and instructions
├── .gitignore                     # Git ignore rules for caches, envs, and artifacts
├── data/
│   ├── sample_transactions.csv    # 5,500+ synthetic transactions covering 24 months
│   └── README.md                  # Dataset schema and data dictionary
├── docs/
│   └── ACADEMIC_REPORT_AND_VIVA.md# Academic report outline and 12+ viva questions
├── models/
│   └── .gitkeep                   # Directory placeholder for serialized artifacts
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_generator.py          # Synthetic transactions generator
│   ├── preprocessing.py           # Schema validation, missing imputation, pipeline builder
│   ├── feature_engineering.py     # Calendar flags, rolling statistics, monthly profiles
│   ├── regression_models.py       # Linear, Polynomial, Ridge/Lasso, Ensemble regression
│   ├── classification_models.py   # Logistic Regression, SVM Kernels, Ensemble classification
│   ├── dimensionality_reduction.py# PCA & LDA algorithms and evaluation
│   ├── clustering_models.py       # K-Means, Elbow method, Silhouette scoring
│   ├── anomaly_detection.py       # DBSCAN density clustering and Z-score anomaly reasoning
│   ├── reinforcement_learning.py  # Environment, Tabular Q-Learning agent, training loop
│   ├── evaluation.py              # Metrics computation and dark-theme Plotly visualizations
│   └── ui_helpers.py              # CSS styling, KPI metric cards, and cached data loader
├── pages/
│   ├── 1_Financial_Overview.py    # Cash flow, monthly trends, and budget adherence
│   ├── 2_Expense_Prediction.py    # Predictive expense modeling and what-if estimator
│   ├── 3_Spending_Analysis.py     # K-Means personas, DBSCAN outliers, PCA vs. LDA
│   ├── 4_Smart_Budget_Planner.py  # Q-Learning budget allocator & simulation
│   └── 5_ML_Laboratory.py         # Dedicated interactive laboratory for all 11 practicals
└── tests/
    ├── test_preprocessing.py      # Unit tests for schema validation and cleaning
    ├── test_models.py             # Unit tests for ML algorithms (P1 to P10)
    └── test_reinforcement_learning.py # Unit tests for Q-Learning Bellman updates & MDP
```

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.11 / 3.14 compatible
- **Data Manipulation:** Pandas, NumPy
- **Scientific & Machine Learning:** Scikit-Learn, SciPy
- **Reinforcement Learning:** NumPy-based Tabular Q-Learning
- **Web Dashboard & UI:** Streamlit
- **Data Visualizations:** Plotly Interactive Graphs, Matplotlib, Seaborn
- **Testing Suite:** Pytest
- **Model Storage:** Joblib

---

## 💻 Installation & Execution Instructions (Windows PowerShell)

Follow these exact steps to set up and launch SpendSmart AI on Windows:

### Step 1: Open Windows PowerShell
Navigate to your project root:
```powershell
# Open Windows Terminal or PowerShell in the project directory
Get-Location
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
If execution policy blocks script activation on Windows, run the bypass command for the current process without altering machine-wide security settings:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Run Automated Verification Tests
Execute the comprehensive Pytest suite:
```powershell
python -m pytest tests/ -v
```
*(All 20 tests should pass cleanly).*

### Step 6: Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📤 Custom CSV Upload Guidelines

You can upload your personal transactions CSV directly in the sidebar. SpendSmart AI requires the following columns:

| Column Name | Expected Format | Description |
|---|---|---|
| `Date` | `YYYY-MM-DD` | Timestamp of transaction |
| `Expense_Amount` | Numeric (e.g. `45.50`) | Cost of transaction (positive float) |
| `Category` | String | e.g. `Groceries`, `Rent`, `Utilities`, `Dining`, etc. |
| `Income` *(Optional)* | Numeric | Monthly user income baseline |
| `Budget` *(Optional)* | Numeric | Allocated category budget limit |
| `Payment_Method` *(Optional)*| String | e.g. `Credit Card`, `Debit Card`, `Cash` |

If optional columns are missing, SpendSmart AI's intelligent imputation engine automatically creates safe default baselines without crashing.

---

## 🛡️ Academic Disclaimer

> **Academic Notice:** SpendSmart AI is developed strictly as an educational software project for engineering students. All expense forecasts, customer cluster labels, anomaly classifications, and Q-Learning budget recommendations are simulated estimates and do **not** constitute certified financial, legal, or investment advice.
