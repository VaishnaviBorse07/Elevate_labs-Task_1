import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
url = "E:/Vaishu coding/Elevate_Labs/Task_3/Housing.csv"
df = pd.read_csv(url)  # Adjust path if needed

# Preview
print("Dataset Preview:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# ---------- SIMPLE LINEAR REGRESSION ----------
# Use 'area' to predict 'price'
X_simple = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

print("\n--- Simple Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R² Score:", r2_score(y_test, y_pred_simple))

# Plot
plt.figure(figsize=(8,6))
sns.regplot(x=X_test['area'], y=y_test, line_kws={'color': 'red'})
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (INR)')
plt.title('Simple Linear Regression - Area vs Price')
plt.show()


# ---------- MULTIPLE LINEAR REGRESSION ----------
# Convert categorical columns using get_dummies
df_encoded = pd.get_dummies(df, drop_first=True)

X_multi = df_encoded.drop(['price'], axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("\n--- Multiple Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, y_pred_multi))
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R² Score:", r2_score(y_test, y_pred_multi))

# Coefficients
coeff_df = pd.DataFrame(model_multi.coef_, X_multi.columns, columns=['Coefficient'])
print("\nFeature Coefficients:\n", coeff_df)



