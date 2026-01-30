import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Title
st.title("🎓 Student Performance Prediction App")
st.write("Predict student exam score based on study habits, lifestyle, and attendance.")

# Load dataset
data = pd.read_csv("student_habits_performance.csv")

# Fill missing parental_education_level with mode
data['parental_education_level'] = data['parental_education_level'].fillna(data['parental_education_level'].mode()[0])

# Features to use
numerical_features = [
    'study_hours_per_day', 'attendance_percentage', 'sleep_hours',
    'social_media_hours', 'exercise_frequency', 'mental_health_rating'
]

categorical_features = [
    'gender', 'diet_quality', 'internet_quality', 'part_time_job', 'extracurricular_participation'
]

X = data[numerical_features + categorical_features]
y = data['exam_score']

# Preprocessing: One-Hot Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough'
)

# Build pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_pipeline.fit(X_train, y_train)

# --- STREAMLIT USER INPUT ---
st.header("Enter Student Details")

study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 85.0)
sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
social_media = st.slider("Social Media Hours per Day", 0.0, 10.0, 2.0)
exercise = st.slider("Exercise Frequency (days/week)", 0, 7, 3)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 7)

gender = st.selectbox("Gender", data['gender'].unique())
diet = st.selectbox("Diet Quality", data['diet_quality'].unique())
internet = st.selectbox("Internet Quality", data['internet_quality'].unique())
part_time = st.selectbox("Part-time Job", data['part_time_job'].unique())
extracurricular = st.selectbox("Extracurricular Participation", data['extracurricular_participation'].unique())

# Combine into dataframe for prediction
new_student = pd.DataFrame([[
    study_hours, attendance, sleep_hours, social_media, exercise, mental_health,
    gender, diet, internet, part_time, extracurricular
]], columns=numerical_features + categorical_features)

# Prediction button
if st.button("Predict Exam Score"):
    prediction = model_pipeline.predict(new_student)
    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")