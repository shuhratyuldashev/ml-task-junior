from pathlib import Path
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH_REGRESSION = SCRIPT_DIR / "../models/random_forest_regression_model.pkl"
MODEL_PATH_CLASSIFICATION = SCRIPT_DIR / "../models/random_forest_classification_model.pkl"

model_random_forest_regression = joblib.load(MODEL_PATH_REGRESSION)
model_random_forest_classification = joblib.load(MODEL_PATH_CLASSIFICATION)



age = int(input("AGE: "))
study_hours_per_week = int(input("STUDY HOURS PER WEEK: "))
attendance_rate = int(input("ATTENDANCE RATE: "))
previous_score = int(input("PREVIOUS SCORE: "))
assignments_completed = int(input("ASSIGNMENTS COMPLETED: "))
sleep_hours = float(input("SLEEP HOURS: "))
practice_tests_taken = int(input("PRACTICE TESTS TAKEN: "))
late_submissions = int(input("LATE SUBMISSIONS: "))
absences = int(input("ABSENCES: "))
internet_usage_hours = float(input("INTERNET USAGE HOURS: "))
social_media_hours = float(input("SOCIAL MEDIA HOURS: "))
gender = input("GENDER: ")
parent_education = input("PARENT EDUCATION: ")
course_type = input("COURSE TYPE: ")
learning_format = input("LEARNING FORMAT: ")
has_part_time_job = input("HAS PART TIME JOB: ")
internet_access = input("INTERNET ACCESS: ")


print(
    """
    ------------------------------------------------
    YOU GONNA TAKE:
    """,
    model_random_forest_regression.predict(
        [[
            age,
            study_hours_per_week,
            attendance_rate,
            previous_score,
            assignments_completed,
            sleep_hours,
            practice_tests_taken,
            late_submissions,
            absences,
            internet_usage_hours,
            social_media_hours,
        ]]
    )
)

print(
    """
    ------------------------------------------------
    UR RISK LEVEL:
    """,
    model_random_forest_classification.predict(
        [[
            age,
            study_hours_per_week,
            attendance_rate,
            previous_score,
            assignments_completed,
            sleep_hours,
            practice_tests_taken,
            late_submissions,
            absences,
            internet_usage_hours,
            social_media_hours,
            gender,
            parent_education,
            course_type,
            learning_format,
            has_part_time_job,
            internet_access,
        ]]
    )
)
