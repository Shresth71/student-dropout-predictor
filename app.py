import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("student_dropout_xgb_final.pkl")  # Make sure this file is in your directory

# Course mapping
course_mapping = {
    "Biofuel Production Technologies": 1,
    "Animation and Multimedia Design": 2,
    "Social Service (evening attendance)": 3,
    "Agronomy": 4,
    "Communication Design": 5,
    "Veterinary Nursing": 6,
    "Informatics Engineering": 7,
    "Equiniculture": 8,
    "Management": 9,
    "Social Service": 10,
    "Tourism": 11,
    "Nursing": 12,
    "Oral Hygiene": 13,
    "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15,
    "Basic Education": 16,
    "Management (evening attendance)": 17
}

# Streamlit UI
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>
        🎓 Student Dropout Risk Predictor
    </h1>
    <p style='text-align: center;'>Enter student academic details and get dropout prediction with risk explanation.</p>
""", unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1, 2.5, 1])

with center_col:
    st.subheader("\U0001F4D8 Student Academic Details")

    course_name = st.selectbox("Select Course", list(course_mapping.keys()))
    total_grade = st.number_input("Total Grade (0.0 – 20.0)", min_value=0.0, max_value=20.0, step=0.1)
    total_approved = st.number_input("Total Approved Subjects (0–40)", min_value=0, max_value=40)
    tuition_up_to_date = st.selectbox("Tuition Fees Up to Date (0 = No, 1 = Yes)", [0, 1])
    scholarship_holder = st.selectbox("Scholarship Holder (0 = No, 1 = Yes)", [0, 1])
    debtor = st.selectbox("Debtor (0 = No, 1 = Yes)", [0, 1])
    age_at_enrollment = st.number_input("Age at Enrollment (15–60)", min_value=15, max_value=60)
    total_enrolled = st.number_input("Total Enrollments (0–15)", min_value=0, max_value=15)
    total_evaluations = st.number_input("Total Evaluations (0–20)", min_value=0, max_value=20)

    if st.button("\U0001F50D Predict Dropout Risk"):
        course_code = course_mapping[course_name]
        input_data = np.array([[total_grade, total_approved, tuition_up_to_date,
                                scholarship_holder, debtor, age_at_enrollment,
                                total_enrolled, total_evaluations, course_code]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        st.subheader("\U0001F4CA Prediction Result")
        if prediction == 1:
            st.error(f"🚨 Likely to Dropout with {probability:.2f}% probability.")
        else:
            st.success(f"✅ Likely to Continue with {100 - probability:.2f}% confidence.")

        st.subheader("\U0001F4A1 Risk Explanation")
        st.markdown("""
        - **Low grades**, **fewer approved subjects**, or **age-related delays** can increase dropout risk.  
        - **Debtor status**, **no scholarship**, or **fee delays** raise concern.  
        - Course choice influences likelihood of persistence.  
        """)

        # 🎯 Add a pie chart showing prediction probability
        labels = ['Continue', 'Dropout']
        sizes = [100 - probability, probability]
        colors = ['#4CAF50', '#F44336']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
