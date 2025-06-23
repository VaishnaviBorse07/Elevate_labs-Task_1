# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Dataset
url = "E:/Vaishu coding/Elevate_Labs/Task_1/Titanic-Dataset.csv"
df = pd.read_csv(url)

# Basic Info
print("Basic Info:\n", df.info())
print("\nSummary Stats:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # Too many nulls
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Encode Categorical Columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Male:1, Female:0

# One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Visualize Outliers
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplot for Age and Fare')
plt.show()

# Optionally remove outliers (e.g., Fare > 3rd quartile + 1.5*IQR)
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

# Final Cleaned Dataset
print("\nFinal Dataset Shape:", df.shape)
print("\nSample:\n", df.head())

# Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)
