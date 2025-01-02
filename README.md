# Data Science Projects: Classification and Regression

## Overview
This repository contains two comprehensive data science projects completed as part of Assignment 2 for the TDSR22 Data Science course. The tasks involve:

1. **Binary Classification**: Predicting whether an individual's income is above or below $50K based on demographic and work-related attributes.
2. **Regression Analysis**: Predicting the median monetary value of houses in California using housing-related attributes.

## Project 1: Binary Classification
### Objective
To classify individuals as having an income either above or below $50K based on a set of demographic and work-related attributes.

### Dataset
- **Attributes**: Age, gender, education, marital status, hours worked per week, etc.
- **Target Variable**: Income (<=50K or >50K).
- **Size**: 48,842 rows.

### Methodology
1. **Data Pre-processing**:
   - Removed rows with missing values.
   - Min-max normalization for numerical features.
   - One-hot encoding for categorical features.
2. **Models Used**:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest Classifier
3. **Evaluation**:
   - 10-fold cross-validation.
   - Metrics: Accuracy, AUC.

### Results
| Model                | Accuracy (%) | AUC   |
|----------------------|--------------|-------|
| Logistic Regression  | 84.61        | 0.904 |
| K-Nearest Neighbors  | 82.81        | 0.881 |
| Random Forest        | 86.05        | 0.892 |

Random Forest was identified as the best-performing model overall.

### Optimization
- **Random Forest**: Increased the number of decision trees to improve accuracy.
- **KNN**: Optimized the number of neighbors using a heuristic approach.

---

## Project 2: Regression Analysis
### Objective
To predict the median monetary value of houses in California using housing attributes.

### Dataset
- **Attributes**: Median age, total rooms, population, median income, etc.
- **Target Variable**: Median house value.
- **Size**: 19,736 rows (after pre-processing).

### Methodology
1. **Data Pre-processing**:
   - Removed noisy data points using DBSCAN clustering.
   - One-hot encoded the `ocean proximity` feature.
2. **Models Used**:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosted Trees (GBT)
3. **Evaluation**:
   - 10-fold cross-validation.
   - Metrics: R², Mean Squared Error (MSE).

### Results
| Model                | R² (%) | MSE    |
|----------------------|---------|--------|
| Linear Regression    | 64.4    | 0.020  |
| Random Forest        | 79.2    | 0.012  |
| Gradient Boosted Trees | 79.7 | 0.012  |

Gradient Boosted Trees performed slightly better and were chosen for optimization.

### Optimization
- **Gradient Boosted Trees**:
  - Learning rate: 0.25
  - Number of models: 450
  - Achieved R² = 85.9% and MSE = 0.009.

---

## How to Use
### Requirements
- Python 3.8+
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, KNIME extensions (for optimization).

### Steps to Reproduce
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ds-classification-regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ds-classification-regression
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebooks for each task:
   - `classification.ipynb`
   - `regression.ipynb`

---

## Acknowledgments
This project was developed by Group #6 (Mahmut Osmanovic, Isac Paulsson, Sebastian Tuura, Mohamed Al Khaled) as part of the TDSR22 Data Science course, 2024.

## License
This repository is licensed under the MIT License. See `LICENSE` for details.
