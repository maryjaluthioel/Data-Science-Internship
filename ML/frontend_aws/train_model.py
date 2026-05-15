import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# 1. Load the dataset
df = pd.read_csv('Amazon_Big_Sales_Dataset_2026.csv')

# 2. Preprocessing
# We extract unique categories for the front-end dropdown later
categories = sorted(df['Category'].unique().tolist())

# Create dummy variables for Category
df_ml = pd.get_dummies(df, columns=['Category'])

# Define Features and Target
# We drop non-numeric/identifier columns
X = df_ml.drop(columns=['Product_ID', 'Product_Name', 'Rating'])
y = df_ml['Rating']

# 3. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save everything required for the app
joblib.dump(model, 'sales_model.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
joblib.dump(categories, 'category_list.joblib')

print("Success: Model, columns, and categories have been saved!")