from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
from pathlib import Path
import json



SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TO_JSON = SCRIPT_DIR / "../input/students.json"

with open(PATH_TO_JSON, "r") as f:
    data = json.load(f)
    
df = pd.DataFrame(data)


y = df["risk_level"]

X = df.drop("risk_level", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_feature = [
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
    "final_score"
]

categorical_features = [
    'gender', 'parent_education', 'course_type', 
    'learning_format', 'has_part_time_job', 'internet_access'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_features)
    ],
    remainder='drop'
)


#Pipeline Random Forest
pipeline_random_forest = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

pipeline_random_forest.fit(X_train, y_train)

#Pipeline Gradient Boosting
pipeline_gradient_boosting = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])

pipeline_gradient_boosting.fit(X_train, y_train)

#Pipeline Logistic Regression
pipeline_logistic_regression = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline_logistic_regression.fit(X_train, y_train)



print("""
===============================================================================
TRAINING CLASSIFICATION MODELS RESULTS:
===============================================================================
""")

print("Random Forest:")
print(f"Accuracy: {accuracy_score(y_test, pipeline_random_forest.predict(X_test))}")
print(f"Classification Report:\n{classification_report(y_test, pipeline_random_forest.predict(X_test))}")
print(f"Precision: {precision_score(y_test, pipeline_random_forest.predict(X_test), average='weighted')}")
print(f"Recall: {recall_score(y_test, pipeline_random_forest.predict(X_test), average='weighted')}")
print(f"F1: {f1_score(y_test, pipeline_random_forest.predict(X_test), average='weighted')}")
print(f"Confusion Matrix: {confusion_matrix(y_test, pipeline_random_forest.predict(X_test))}")
print("----------------------------------------------------------------")
print("Gradient Boosting:")
print(f"Accuracy: {accuracy_score(y_test, pipeline_gradient_boosting.predict(X_test))}")
print(f"Classification Report:\n{classification_report(y_test, pipeline_gradient_boosting.predict(X_test))}")
print(f"Precision: {precision_score(y_test, pipeline_gradient_boosting.predict(X_test), average='weighted')}")
print(f"Recall: {recall_score(y_test, pipeline_gradient_boosting.predict(X_test), average='weighted')}")
print(f"F1: {f1_score(y_test, pipeline_gradient_boosting.predict(X_test), average='weighted')}")
print(f"Confusion Matrix: {confusion_matrix(y_test, pipeline_gradient_boosting.predict(X_test))}")
print("----------------------------------------------------------------")
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(y_test, pipeline_logistic_regression.predict(X_test))}")
print(f"Classification Report:\n{classification_report(y_test, pipeline_logistic_regression.predict(X_test))}")
print(f"Precision: {precision_score(y_test, pipeline_logistic_regression.predict(X_test), average='weighted')}")
print(f"Recall: {recall_score(y_test, pipeline_logistic_regression.predict(X_test), average='weighted')}")
print(f"F1: {f1_score(y_test, pipeline_logistic_regression.predict(X_test), average='weighted')}")
print(f"Confusion Matrix: {confusion_matrix(y_test, pipeline_logistic_regression.predict(X_test))}")



joblib.dump(pipeline_random_forest, SCRIPT_DIR / "../models/random_forest_classification_model.pkl")
