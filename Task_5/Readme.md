# 🏡 Task 5: Decision Tree & Random Forest (Housing Dataset)

## 📌 Objective
Train and compare Decision Tree and Random Forest models on housing price prediction using the Housing.csv dataset.

---

## 📁 Dataset
- **File**: Housing.csv
- **Source**: Internship-provided dataset
- **Target Variable**: `price`

---

## 🧰 Tools Used
- Python 3, pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## 🔍 Steps Performed
1. Encoded categorical columns like `mainroad`, `furnishingstatus`, etc.
2. Split the data into training and test sets.
3. Trained a **Decision Tree Regressor** with depth control and visualized it.
4. Evaluated using R² and MSE.
5. Trained a **Random Forest Regressor**.
6. Compared results and plotted **feature importance**.

---

## 📊 Results

| Model              | R² Score | Cross-Val R² |
|--------------------|----------|---------------|
| Decision Tree      | ~0.65    | ~0.61         |
| Random Forest      | ~0.85    | ~0.81         |

Random Forest clearly outperformed due to better generalization and ensemble averaging.

---

## 📈 Visuals
### 🧠 Decision Tree:
![Decision Tree](tree_visualization.png)

### 📌 Feature Importance:
![Feature Importance](feature_importance.png)

---

## 💡 Concepts Covered
- Regression Trees
- Overfitting & Tree Depth
- Ensemble Learning with Bagging
- Feature Importance

---

## 📤 Submission
Upload your repo with:
- `task5_model.py`
- `Housing.csv`
- Generated plots
- `README.md`
