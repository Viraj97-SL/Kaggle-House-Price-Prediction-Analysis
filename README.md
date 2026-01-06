
# House Price Prediction Analysis 🏠

This repository contains a comprehensive data science workflow to predict housing prices based on the famous [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

The project demonstrates the end-to-end process of a machine learning project, from exploratory data analysis (EDA) to ensemble modeling.

## 📌 Project Overview

The goal of this analysis is to predict the final sale price of homes in Ames, Iowa, using 79 explanatory variables describing (almost) every aspect of residential homes.

### Key Highlights
* **Data Preprocessing**: Handling missing values, outlier detection, and feature engineering.
* **Statistical Analysis**: Addressing target variable skewness using Log Transformation ($\log(1+x)$) to normalize the distribution.
* **Modeling**: Implementation of regularized linear regression models (**Lasso**, **Ridge**) and Gradient Boosting (**XGBoost**).
* **Ensemble Learning**: A final hybrid model averaging predictions from Lasso and XGBoost to minimize variance and improve the Root Mean Squared Logarithmic Error (RMSLE).

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas & NumPy** (Data manipulation)
* **Matplotlib & Seaborn** (Data visualization)
* **Scikit-Learn** (Preprocessing, Lasso, Ridge, KFold validation)
* **SciPy** (Statistical analysis, Skew, Norm)
* **XGBoost** (Gradient boosting)

## 📂 Notebook Structure

The analysis in `House_Price_Prediction_Analysis.ipynb` follows these steps:

1.  **Load and Combine Data**: Reading `train.csv` and `test.csv` and combining them for consistent preprocessing.
2.  **EDA (Exploratory Data Analysis)**: Analyzing distributions, correlations, and data quality.
3.  **Target Variable Analysis**: Applying log-transformation to `SalePrice` to correct right-skewness.
4.  **Feature Engineering**: Encoding categorical variables and scaling numerical features.
5.  **Model Building**:
    * *Ridge Regression (L2)*: Handles multicollinearity.
    * *Lasso Regression (L1)*: Performs feature selection.
    * *XGBoost*: Captures non-linear relationships.
6.  **Ensemble Prediction**: Averaging the outputs of Lasso (50%) and XGBoost (50%) for the final result.

## 🚀 Getting Started

### Prerequisites
Ensure you have the necessary libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost

```

### Usage

1. Clone this repository.
2. Download the `train.csv` and `test.csv` files from Kaggle and place them in the root directory.
3. Run the Jupyter Notebook:

```bash
jupyter notebook "House_Price_Prediction_Analysis (3).ipynb"

```

## 📊 Results

The ensemble approach combines the stability of linear models with the power of gradient boosting. The final predictions are exponentiated (`np.expm1`) to reverse the initial log transformation, providing results in the original currency scale.

---

*Author: [Viraj Bulugahapitiya/Viraj97-SL]*

```

```
