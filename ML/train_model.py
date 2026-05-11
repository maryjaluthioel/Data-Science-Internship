
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the data
try:
    df = pd.read_csv('titanic.csv')
    print("File loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

# 2. Data Cleaning (Handling missing values)
# Based on your notebook, we know Age and Embarked have missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Feature Engineering
# Convert Categorical strings to Numbers for the model
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 4. Selecting Features
# We exclude PassengerId, Name, Ticket, and Cabin for this basic model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. Validation
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc * 100:.2f}%")

# 8. Save the model using joblib
# This file 'titanic_model.pkl' is what Streamlit will use
joblib.dump(model, 'titanic_model.pkl')
print("Model saved as titanic_model.pkl")