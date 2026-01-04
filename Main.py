
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load and Prepare Data
@st.cache_data
def load_data():
    df = pd.read_csv('your_data.csv') # Replace with your filename
    # Encoding categorical data (Gender, Risk Level)
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    # Store the mapping for the target
    target_le = LabelEncoder()
    df['risk_level'] = target_le.fit_transform(df['risk_level'])
    return df, target_le

df, target_le = load_data()

# 2. Train the Model
X = df.drop('risk_level', axis=1)
y = df['risk_level']
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. The Web Interface (Streamlit)
st.title("🌱 MindCheck: Student AI Support")
st.write("Daily check-in for mental well-being.")

# Create inputs for the user
age = st.number_input("Age", 10, 20, 15)
sleep = st.slider("Sleep Hours", 0, 12, 8)
screen = st.slider("Screen Time (Hours)", 0, 12, 4)
stress = st.slider("Current Stress Level (1-10)", 1, 10, 5)
# ... add other inputs based on your CSV columns ...

if st.button("Analyze My Well-being"):
    # Predict
    # Note: Ensure the order of features matches your X training data
    user_data = [[age, 0, sleep, screen, 5, stress, 30, 5, 90, 5]] # Example dummy array
    prediction = model.predict(user_data)
    risk = target_le.inverse_transform(prediction)[0]
    
    # 4. Explainable AI Component
    st.subheader(f"Risk Assessment: {risk}")
    
    if risk == "High":
        st.warning("Our AI detects patterns similar to burnout. Please consider speaking with a school counselor.")
        st.info("💡 Suggestion: Try the '5-4-3-2-1' grounding technique and reduce screen time by 1 hour today.")
    else:
        st.success("You're doing great! Keep maintaining your current routine.")
