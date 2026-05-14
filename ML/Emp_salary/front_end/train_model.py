import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your generated dataset
df = pd.read_csv('employee_data.csv')

# Preprocessing
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_dept = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education'] = le_edu.fit_transform(df['Education'])
df['Department'] = le_dept.fit_transform(df['Department'])

X = df.drop(['Employee_ID', 'Salary', 'Salary_Category'], axis=1)
y = df['Salary']

# Train the best model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# SAVE EVERYTHING
joblib.dump(model, 'salary_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_edu, 'le_edu.pkl')
joblib.dump(le_dept, 'le_dept.pkl')

print("Models and Encoders saved successfully!")