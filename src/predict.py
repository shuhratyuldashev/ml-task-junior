import pandas as pd
from pathlib import Path
import joblib

from recommendations import get_recommendation
from validation import validate_student

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH_REGRESSION = SCRIPT_DIR / "../models/score_regression_model.pkl" # Убедись, что имена файлов правильные
MODEL_PATH_CLASSIFICATION = SCRIPT_DIR / "../models/risk_classification_model.pkl"

model_regression = joblib.load(MODEL_PATH_REGRESSION)
model_classification = joblib.load(MODEL_PATH_CLASSIFICATION)

print("=== ВВОД ДАННЫХ СТУДЕНТА ===")
student = {
    "age": int(input("AGE: ")),
    "study_hours_per_week": int(input("STUDY HOURS PER WEEK: ")),
    "attendance_rate": int(input("ATTENDANCE RATE: ")),
    "previous_score": int(input("PREVIOUS SCORE: ")),
    "assignments_completed": int(input("ASSIGNMENTS COMPLETED: ")),
    "sleep_hours": float(input("SLEEP HOURS: ")),
    "practice_tests_taken": int(input("PRACTICE TESTS TAKEN: ")),
    "late_submissions": int(input("LATE SUBMISSIONS: ")),
    "absences": int(input("ABSENCES: ")),
    "internet_usage_hours": float(input("INTERNET USAGE HOURS: ")),
    "social_media_hours": float(input("SOCIAL MEDIA HOURS: ")),
    "gender": input("GENDER (male/female): "),
    "parent_education": input("PARENT EDUCATION: "),
    "course_type": input("COURSE TYPE: "),
    "learning_format": input("LEARNING FORMAT: "),
    "has_part_time_job": input("HAS PART TIME JOB (yes/no): "),
    "internet_access": input("INTERNET ACCESS (yes/no): ")
}

try:
    validate_student(student)
except ValueError as e:
    print(f"Ошибка валидации: {e}")
    exit()

student_df = pd.DataFrame([student])


predicted_score = model_regression.predict(student_df)[0]
risk_level = model_classification.predict(student_df)[0]

print("\n------------------------------------------------")
print(f"UR PREDICTED SCORE: {predicted_score:.1f}")
print(f"UR RISK LEVEL: {risk_level.upper()}")
print("------------------------------------------------")

recommendation = get_recommendation(student, predicted_score, risk_level)
print(f"RECOMMENDATION:\n{recommendation}")