import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os 
from pathlib import Path
import json


SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TO_JSON = SCRIPT_DIR / "../input/students.json"

with open(PATH_TO_JSON, "r") as f:
    data = json.load(f)
    
df = pd.DataFrame(data)
    
    
X = df[
    [
        "age", 
        "study_hours_per_week", 
        "attendance_rate", 
        "previous_score", 
        "assignments_completed", 
        "sleep_hours", 
        "practice_tests_taken", 
        "late_submissions", 
        "absences", 
        "internet_usage_hours", 
        "social_media_hours", 
    ]
]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest
model_random_forest = RandomForestRegressor()

#Gradient Boosting
model_gradient_boosting = GradientBoostingRegressor()

#Linear Regression
model_linear_regression = LinearRegression()


#Train models 
model_random_forest.fit(X_train, y_train)
model_gradient_boosting.fit(X_train, y_train)
model_linear_regression.fit(X_train, y_train)
 
#Test models 
y_pred_random_forest = model_random_forest.predict(X_test)
y_pred_gradient_boosting = model_gradient_boosting.predict(X_test)
y_pred_linear_regression = model_linear_regression.predict(X_test)

#Results models
print("---------------")
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_random_forest))
print("Random Forest R2:", r2_score(y_test, y_pred_random_forest))
print("---------------")
print("Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_gradient_boosting))
print("Gradient Boosting R2:", r2_score(y_test, y_pred_gradient_boosting))
print("---------------")
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear_regression))
print("Linear Regression R2:", r2_score(y_test, y_pred_linear_regression))
print("---------------")

#Save models
joblib.dump(model_random_forest, SCRIPT_DIR / "../models/random_forest_regression_model.pkl")


