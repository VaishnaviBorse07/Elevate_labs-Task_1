# 🎓 EDA: Student Performance Dataset

## 📌 Objective
This project performs **Exploratory Data Analysis (EDA)** on the *Student Performance Dataset*, exploring how factors like gender, lunch, and test preparation affect student scores in Math, Reading, and Writing.

## 📁 Dataset
- **File**: `student_performance.csv`

### Features:
- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `math score`
- `reading score`
- `writing score`

---

## 🧰 Tools Used
- Python
- Spyder IDE
- Pandas
- Matplotlib
- Seaborn

---

## 🧪 EDA Performed
- Displayed first few rows, info, and summary statistics
- Checked for missing values
- Plotted:
  - Histograms of all 3 scores
  - Boxplots of scores to detect outliers
  - Gender-wise Math score distribution
  - Correlation heatmap between scores
  - Pairplot for score relationships

---

## 📈 Key Insights
- **High Correlation** between Reading and Writing scores.
- **Males** slightly outperform females in **Math**, while **females** tend to score better in Reading and Writing.
- **Students who completed** the test preparation course scored **higher on average** in all subjects.

---

## 🗂️ Folder Structure

EDA_Student_Performance/
├── student_eda.py # Python file with full EDA code (for Spyder)
├── student_performance.csv # Dataset file
└── README.md # This file

---

## 🚀 How to Run
1. Open `student_eda.py` in Spyder.
2. Ensure `student_performance.csv` is in the same folder.
3. Run the script. Output will appear in the Console, and plots in the Plots tab.

---

## ✅ Submission
Push this project to a GitHub repository and submit the repository link in the required submission form.

---