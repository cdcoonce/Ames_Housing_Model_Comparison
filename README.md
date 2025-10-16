# 🏠 Ames Housing Price Prediction — Baselines vs Neural Network

**Author:** Charles Coonce  
**Course:** DAT 402 — Applied Predictive Modeling  
**Date:** October 2025  

---

## 🎯 Project Overview

This project analyzes the **Ames Housing Dataset**, a well-known dataset for regression tasks in predictive modeling.  
The goal was to **compare three models** — Linear Regression, k-Nearest Neighbors (kNN), and a Neural Network — to understand the **bias–variance tradeoff** and observe how model flexibility impacts predictive performance.

---

## 📊 Objectives

- Build and evaluate **Linear Regression** and **kNN** baselines  
- Demonstrate **bias–variance tradeoff** and **cross-validation**  
- Develop a **Neural Network** using TensorFlow/Keras  
- Compare model accuracy using RMSE, MAE, and R² metrics  
- Export results and analysis to an interactive HTML notebook  

---

## 📦 Dataset

**Source:** [Kaggle — Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

- **Instances:** 2,930 homes  
- **Features:** 79 (numerical + categorical)  
- **Target Variable:** `SalePrice`  
- **Key Engineered Features:**
  - `TotalSF` — total living area (basement + first + second floor)  
  - `TotalBath` — combined count of full and half baths  
  - `AgeAtSale` — house age when sold  
  - `RemodelAge` — time since last remodel  

---

## 🧰 Tools & Libraries

| Category | Libraries Used |
|-----------|----------------|
| Data Handling | `pandas`, `numpy`, `matplotlib` |
| Modeling | `scikit-learn` (`LinearRegression`, `KNeighborsRegressor`) |
| Neural Network | `TensorFlow`, `Keras` |
| Preprocessing | `ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder` |
| Validation & Metrics | `train_test_split`, `KFold`, `cross_val_score`, `RMSE`, `MAE`, `R²` |

---

## ⚙️ Methodology

1. **Data Preparation**
   - Removed irrelevant columns, handled missing data.
   - Selected top correlated features with `SalePrice`.
   - Added engineered features to improve interpretability.

2. **Preprocessing Pipeline**
   - Imputation for numeric and categorical features.
   - Standard scaling for numeric data.
   - One-hot encoding for categorical variables.

3. **Model Training**
   - Linear Regression: baseline with high bias, low variance.
   - kNN: explored `k = 1–40` to demonstrate bias–variance tradeoff.
   - Neural Network: two hidden layers (128 → 64 ReLU), linear output.

4. **Evaluation Metrics**
   - **RMSE** — Root Mean Squared Error  
   - **MAE** — Mean Absolute Error  
   - **R²** — Coefficient of Determination  

---

## 📈 Results Summary

| Model | RMSE | MAE | Notes |
|--------|------|-----|-------|
| **Linear Regression** | 28,152 | 19,184 | Simple, interpretable baseline |
| **kNN (k=5)** | 27,025 | 18,276 | Improved accuracy, more flexible |
| **Neural Network** | **26,940** | **18,451** | Best performance; captured complex relationships |

- The **Neural Network** achieved the lowest overall error and best generalization.  
- **kNN** demonstrated a clear bias–variance tradeoff.  
- **Linear Regression** remained useful as a transparent baseline.

---

## 🧠 Key Takeaways

- Increasing model flexibility (from Linear → kNN → NN) **reduced bias and improved accuracy**.
- The **Neural Network** achieved balanced performance without significant overfitting.
- **Cross-validation** confirmed kNN’s advantage over Linear Regression.
- Proper preprocessing and feature engineering significantly enhanced results.

---

## 🪄 Visualizations

- **Correlation heatmap** to identify predictive features  
- **kNN bias–variance plot** showing RMSE vs. neighbors  
- **Neural network training curves** (RMSE & MAE over epochs)  
- **Prediction parity plot** — predicted vs. actual prices  
*(All plots are viewable in the accompanying Jupyter Notebook or HTML export.)*

---

## 💡 Future Improvements

- Perform **hyperparameter tuning** for the Neural Network  
- Add **dropout** or **regularization** to reduce potential overfitting  
- Experiment with **ensemble methods** like Random Forest or XGBoost  
- Extend analysis with **feature importance** and **SHAP values**  

---

## 🧩 Repository Structure

```text

Ames_Housing_Price_Prediction/
│
├── data/
│   └── AmesHousing.csv
│
├── notebooks/
│   └── AmesHousing_Project2.ipynb
│
├── exports/
│   └── AmesHousing_Project2.html
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run (Using uv)

```bash
# 1️⃣ Clone the repository
git clone https://github.com/cdcoonce/Ames_Housing_Model_Comparison.git
cd Ames_Housing_Model_Comparison

# 2️⃣ Create a uv-managed virtual environment (Python 3.11 recommended)
uv venv --python 3.11
source .venv/bin/activate

# 3️⃣ Install project dependencies
uv sync
```

---

**📬 Author:**  
*Charles Coonce*  
[Portfolio Website](https://charleslikesdata.com) · [GitHub](https://github.com/cdcoonce)
