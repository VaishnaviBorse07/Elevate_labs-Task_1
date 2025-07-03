import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Standardize the Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train SVM (Linear Kernel)
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# 5. Train SVM (RBF Kernel)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# 6. Evaluate Both Models
print("----- Linear Kernel Evaluation -----")
print(classification_report(y_test, svm_linear.predict(X_test)))

print("----- RBF Kernel Evaluation -----")
print(classification_report(y_test, svm_rbf.predict(X_test)))

# 7. Hyperparameter Tuning (RBF)
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("----- Best Parameters from GridSearch -----")
print(grid.best_params_)

# 8. Visualize Decision Boundary using First 2 Features
def plot_decision_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Use only 2 features for visualization
X_vis = X[:, :2]
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis, y, test_size=0.2, random_state=42)

clf_vis = SVC(kernel='rbf', C=1.0, gamma='scale')
clf_vis.fit(X_vis_train, y_vis_train)

plot_decision_boundary(clf_vis, X_vis_test, y_vis_test, "RBF Kernel Decision Boundary (2D Features)")
