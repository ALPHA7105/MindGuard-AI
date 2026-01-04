import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="MindGuard AI", page_icon="🧠", layout="wide")

# --- DATA PREPARATION & MODEL TRAINING ---
@st.cache_resource
def train_model():
    # Load your dataset
    df = pd.read_csv('mindguard_ai_dataset.csv')
    
    # Encoding categorical columns
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_mood = LabelEncoder()
    df['mood_level'] = le_mood.fit_transform(df['mood_level'])
    
    le_risk = LabelEncoder()
    df['risk_level'] = le_risk.fit_transform(df['risk_level'])
    
    # Features and Target
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_gender, le_mood, le_risk, X.columns

model, le_gender, le_mood, le_risk, feature_names = train_model()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Daily Wellness Check-in")

# Adding 'key' to each widget allows the reset button to find and clear them
age = st.sidebar.number_input("Age", 13, 18, 15, key="age_val")
grade = st.sidebar.selectbox("Grade", [8, 9, 10, 11], key="grade_val")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender_val")
sleep = st.sidebar.slider("Sleep Hours last night", 0.0, 12.0, 7.0, key="sleep_val")
screen = st.sidebar.slider("Screen Time (Hours)", 0.0, 12.0, 4.0, key="screen_val")
academic = st.sidebar.slider("Academic Load (1-5)", 1, 5, 3, key="acad_val")
stress = st.sidebar.slider("Stress Level (1-10)", 1.0, 10.0, 5.0, key="stress_val")
physical = st.sidebar.number_input("Physical Activity (Minutes)", 0, 180, 30, key="phys_val")
social = st.sidebar.slider("Social Interaction Level (1-5)", 1, 5, 3, key="social_val")
attendance = st.sidebar.slider("Attendance %", 50, 100, 90, key="att_val")
mood = st.sidebar.selectbox("Current Mood", ["Low", "Neutral", "Good"], key="mood_val")

st.sidebar.divider()

# The Reset Button
if st.sidebar.button("🔄 Reset All Fields"):
    st.session_state.clear()
    st.rerun()

# The Analyze Button
analyze_btn = st.sidebar.button("Analyze My Well-being")

# --- MAIN PAGE ---
st.title("🧠 MindGuard AI: Student Mental Health Support")
st.markdown("""
**Goal:** Early detection of mental health risks using AI to promote **SDG 3: Good Health & Well-being**.
This tool provides personalized coping strategies and is *not* a medical diagnosis.
""")

if st.sidebar.button("Analyze My Well-being"):
    # Encode inputs
    gender_enc = 1 if gender == "Male" else 0
    mood_map = {"Low": 0, "Neutral": 1, "Good": 2} # Matches common label encoding
    mood_enc = mood_map[mood]
    
    # Create input array for prediction
    user_input = np.array([[age, grade, gender_enc, sleep, screen, academic, 
                            stress, physical, social, attendance, mood_enc]])
    
    # Prediction
    prediction = model.predict(user_input)
    risk_res = le_risk.inverse_transform(prediction)[0]
    
    # Display Result
    st.divider()
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Your Risk Assessment")
        if risk_res == "Low":
            st.success(f"Risk Level: {risk_res}")
            st.write("You seem to be managing well! Keep up your healthy habits.")
        elif risk_res == "Moderate":
            st.warning(f"Risk Level: {risk_res}")
            st.write("You might be feeling some pressure. It’s a good time to practice self-care.")
        else:
            st.error(f"Risk Level: {risk_res}")
            st.write("Our AI detects patterns associated with high stress or burnout. We recommend talking to a trusted adult or school counselor.")

    # --- EXPLAINABLE AI (XAI) ---
    with cols[1]:
        st.subheader("Explainable AI: Why this result?")
        # Show feature importance from the model
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(5)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette="viridis", ax=ax)
        st.pyplot(fig)
        st.caption("The graph shows which factors contributed most to the AI's decision.")

    # --- SDG 3 ALIGNED COPING STRATEGIES ---
    st.divider()
    st.subheader("💡 Personalized Coping Strategies")
    
    advice_cols = st.columns(3)
    with advice_cols[0]:
        st.info("**Sleep & Rest**")
        if sleep < 7:
            st.write("- Aim for 8 hours of sleep to improve cognitive function.")
        else:
            st.write("- Great job on sleep! Consistency is key.")
            
    with advice_cols[1]:
        st.info("**Stress Management**")
        if stress > 7:
            st.write("- Try the 5-4-3-2-1 grounding technique.")
            st.write("- Break your academic tasks into 25-minute 'Pomodoro' chunks.")
        else:
            st.write("- Keep practicing mindfulness to maintain these levels.")
            
    with advice_cols[2]:
        st.info("**Social & Physical**")
        if physical < 30:
            st.write("- Even a 10-minute walk can boost your endorphins.")
        if social < 3:
            st.write("- Reach out to a friend today; social connection reduces anxiety.")

else:
    st.info("Fill in the details on the left and click 'Analyze My Well-being' to start.")

# --- ETHICS & FOOTER ---
st.divider()
st.caption("🔒 **Privacy & Ethics:** All data is processed locally for this demo. No personal identification is stored. This is an AI prototype for educational purposes (SDG 3.4).")
