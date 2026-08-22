import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(page_title="Nadia Traffic AI Predictor", page_icon="🚦", layout="centered")

st.title("🚦 AI Traffic Predictor — Nadia Corridors")
st.markdown("Predict congestion levels in **Bethuadahari (NH-12)** & **Mayapur (Hulor Ghat)** using Machine Learning.")

# Train ML model in background
@st.cache_resource
def load_trained_model():
    np.random.seed(42)
    num_samples = 1000
    locations = np.random.choice([0, 1], size=num_samples)
    hours = np.random.randint(0, 24, size=num_samples)
    is_festival = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
    weather = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])

    train_passing, congestion_levels = [], []
    for loc, hr, fest, weath in zip(locations, hours, is_festival, weather):
        is_train = np.random.choice([0, 1], p=[0.7, 0.3]) if loc == 0 else 0
        train_passing.append(is_train)
        if loc == 0 and is_train == 1:
            cong = 2
        elif loc == 1 and fest == 1:
            cong = 2
        elif hr in [8, 9, 10, 17, 18, 19] and (weath == 1 or fest == 1):
            cong = 2
        elif hr in [8, 9, 10, 17, 18, 19] or weath == 1:
            cong = 1
        else:
            cong = 0
        congestion_levels.append(cong)

    df = pd.DataFrame({
        'Location': locations, 'Hour': hours, 'Is_Festival': is_festival,
        'Train_Passing': train_passing, 'Weather': weather, 'Congestion': congestion_levels
    })
    
    X = df[['Location', 'Hour', 'Is_Festival', 'Train_Passing', 'Weather']]
    y = df['Congestion']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = load_trained_model()

# User input form
st.subheader("📋 Enter Journey Details")

route = st.selectbox("Select Corridor Route:", ["Bethuadahari (NH-12 Level Crossing)", "Mayapur (Hulor Ghat Road)"])
hour = st.slider("Hour of the Day (24-Hour Format):", 0, 23, 9)
festival = st.radio("Is today a major festival day?", ["No", "Yes"], horizontal=True)
gate_closed = st.radio("Is the railway gate closed? (Bethuadahari Only)", ["No", "Yes"], horizontal=True)
weather = st.selectbox("Current Weather:", ["Clear", "Rain"])

if st.button("🚀 Predict Traffic Status"):
    loc_val = 0 if "Bethuadahari" in route else 1
    fest_val = 1 if festival == "Yes" else 0
    train_val = 1 if (gate_closed == "Yes" and loc_val == 0) else 0
    weather_val = 1 if weather == "Rain" else 0

    input_df = pd.DataFrame([[loc_val, hour, fest_val, train_val, weather_val]], 
                            columns=['Location', 'Hour', 'Is_Festival', 'Train_Passing', 'Weather'])
    
    pred = model.predict(input_df)[0]

    st.markdown("---")
    if pred == 0:
        st.success("🟢 **LOW CONGESTION**: Route is completely clear. Safe to travel!")
    elif pred == 1:
        st.warning("🟡 **MEDIUM CONGESTION**: Moderate delays expected (15–20 min buffer suggested).")
    else:
        if loc_val == 0:
            st.error("🔴 **SEVERE GRIDLOCK**: Railway gate active on NH-12! Delay departure by 25–30 minutes.")
        else:
            st.error("🔴 **SEVERE GRIDLOCK**: Hulor Ghat / Temple road choked! Take alternative ferry via Swarupganj.")