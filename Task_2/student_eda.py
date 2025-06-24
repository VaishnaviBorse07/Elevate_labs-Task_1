# student_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("E:/Vaishu coding/Elevate_Labs/Task_2/StudentsPerformance.csv")

# Display first few rows
print("First 5 Rows:\n", df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive statistics
print("\nSummary Statistics:\n", df.describe())

# Histogram for score distributions
df[['math score', 'reading score', 'writing score']].hist(bins=15, figsize=(10, 5))
plt.suptitle("Distribution of Scores")
plt.tight_layout()
plt.show()

# Boxplot of all scores
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
plt.title("Boxplot of Scores")
plt.show()

# Boxplot: Math score by gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Math Scores by Gender")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['math score', 'reading score', 'writing score']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Scores")
plt.show()

# Pairplot for scores
sns.pairplot(df[['math score', 'reading score', 'writing score']])
plt.suptitle("Pairplot of Scores", y=1.02)
plt.show()
