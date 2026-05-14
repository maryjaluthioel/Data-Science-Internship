import pandas as pd
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Train SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train_scaled, y_train)

# 4. Save the model and scaler to files
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully!")