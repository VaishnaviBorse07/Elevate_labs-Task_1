# 🚢 Titanic Data Cleaning & Preprocessing

This repository contains the implementation for **Task 1** of the AI & ML Internship — focused on **Data Cleaning and Preprocessing** using the **Titanic Dataset**.

---

## 🔍 What I Did

I cleaned and prepared the Titanic dataset for machine learning by:
- Exploring the data and handling missing values
- Encoding categorical variables (`Sex`, `Embarked`)
- Scaling numerical features (`Age`, `Fare`)
- Detecting and removing outliers
- Saving the cleaned dataset for future model building

---

## 📌 Objective

To learn and apply data preprocessing techniques including:
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Outlier detection and removal

---

## 🛠 Tools & Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 📂 Dataset

- Dataset used: [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) 

---

## ✅ Steps Performed

1. **Loaded** the Titanic dataset and explored the structure using `.info()`, `.describe()`, and `.isnull().sum()`.
2. **Handled missing values** in `Age` and `Embarked` using median and mode respectively. Dropped the `Cabin` column due to excessive nulls.
3. **Encoded categorical features**:
   - `Sex` was label-encoded (Male: 1, Female: 0)
   - `Embarked` was one-hot encoded
4. **Scaled numerical features** (`Age`, `Fare`) using `StandardScaler`.
5. **Visualized outliers** using `boxplot` and removed extreme outliers in `Fare` using IQR method.
6. **Saved** the cleaned dataset as `titanic_cleaned.csv`.

---

## 📁 Files Included

- `Preprocessing.py` – Main Python script for preprocessing
- `titanic_cleaned.csv` – Cleaned output data
- `README.md` – Description and documentation of the task

## 🙋‍♀️ Author

**Vaishnavi Borse**  
AI & ML Intern  

---
