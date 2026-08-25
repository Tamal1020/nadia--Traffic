import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Nadia AI Traffic Predictor",
    page_icon="🚦",
    layout="centered"
)

# ----------------- CSS STYLING -----------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        height: 3.2em;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- TRAIN DYNAMIC ML MODEL -----------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 6000

    corridor = np.random.choice([0, 1], size=n_samples) # 0: Bethuadahari, 1: Mayapur
    hour = np.random.randint(0, 24, size=n_samples)
    
    # Festivals: 0: None, 1: Gaura Purnima, 2: Jhulan Yatra, 3: Janmashtami / Ratha Yatra, 
    # 4: Durga Puja / Kali Puja, 5: Jagadhatri / Rash Mela
    festival_type = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.55, 0.10, 0.08, 0.09, 0.10, 0.08])
    railway_gate = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
    weather = np.random.choice([0, 1, 2], size=n_samples, p=[0.60, 0.25, 0.15]) # 0: Clear, 1: Rain, 2: Fog

    score = np.zeros(n_samples)

    for i in range(n_samples):
        s = 10
        h = hour[i]
        c = corridor[i]
        f = festival_type[i]

        # 1. BASE TIME EFFECT
        if 8 <= h <= 11:
            s += 30  # Morning Peak
        elif 17 <= h <= 21:
            s += 35  # Evening Peak
        elif 0 <= h <= 5:
            s -= 15  # Late night clearance
        else:
            s += 10  # Normal Daytime

        # 2. SPECIFIC FESTIVAL & TIME DYNAMICS
        if f != 0:
            if f == 1: # Gaura Purnima (Mayapur epic jam all day)
                s += 50 if (7 <= h <= 23) else 25
            elif f == 2: # Jhulan Yatra (Mayapur evening swing ceremony)
                s += 45 if (16 <= h <= 22) else 15
            elif f == 3: # Janmashtami / Ratha Yatra
                s += 45 if (15 <= h <= 23) else 20
            elif f == 4: # Durga Puja / Kali Puja (Bethuadahari NH-12 night pandal hopping)
                s += 45 if (17 <= h <= 24 or 0 <= h <= 2) else 15
            elif f == 5: # Jagadhatri Puja / Rash Mela
                s += 40 if (16 <= h <= 23) else 15

        # 3. RAILWAY GATE DYNAMICS (Bethuadahari NH-12)
        if c == 0 and railway_gate[i] == 1:
            s += 40

        # 4. WEATHER DYNAMICS
        if weather[i] == 1: # Heavy Rain
            s += 20
        elif weather[i] == 2: # Fog
            s += 15

        score[i] = s

    # Assign 4 Target Congestion Classes
    congestion = []
    for s in score:
        if s < 30:
            congestion.append(0) # 0: Low / Smooth
        elif s < 55:
            congestion.append(1) # 1: Moderate
        elif s < 80:
            congestion.append(2) # 2: High Congestion
        else:
            congestion.append(3) # 3: Critical / Severe Jam

    df = pd.DataFrame({
        'corridor': corridor,
        'hour': hour,
        'festival_type': festival_type,
        'railway_gate': railway_gate,
        'weather': weather,
        'congestion': congestion
    })

    X = df[['corridor', 'hour', 'festival_type', 'railway_gate', 'weather']]
    y = df['congestion']

    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model()

# ----------------- UI INTERFACE -----------------
st.title("🚦 Nadia AI Traffic & Festival Predictor")
st.caption("Machine Learning prediction for NH-12 & Hulor Ghat Ghat ferry corridors with live festival dynamics.")

st.markdown("---")
st.subheader("📍 Journey & Route Configuration")

corridor_map = {
    "Mayapur (Hulor Ghat Ferry & Temple Approach)": 1,
    "Bethuadahari (NH-12 Junction & Level Crossing)": 0
}
selected_corridor_label = st.selectbox("Select Corridor:", list(corridor_map.keys()))
selected_corridor = corridor_map[selected_corridor_label]

hour = st.slider("Select Time of Travel (24-Hour Clock):", min_value=0, max_value=23, value=18, 
                 help="Slide to see how morning, afternoon, evening rush, and late night change the AI prediction.")

# Route-Specific Festivals
if selected_corridor == 1: # Mayapur
    festival_choices = {
        "Normal Regular Day (No Festival)": 0,
        "Jhulan Yatra (Evening Swing Darshan)": 2,
        "Gaura Purnima (Mega International Festival)": 1,
        "Sri Krishna Janmashtami / Ratha Yatra": 3,
        "Nabadwip / Mayapur Rash Mela": 5
    }
else: # Bethuadahari
    festival_choices = {
        "Normal Regular Day (No Festival)": 0,
        "Durga Puja (Pandal Hopping & NH-12 Rush)": 4,
        "Kali Puja / Diwali": 4,
        "Jagadhatri Puja (Krishnanagar-Bethua Belt)": 5,
        "Eid-ul-Fitr / Regional Mela": 3
    }

selected_fest_label = st.selectbox("Select Festival / Occasion:", list(festival_choices.keys()))
selected_festival = festival_choices[selected_fest_label]

if selected_corridor == 0:
    gate_input = st.radio("Is Bethuadahari Railway Gate Closed?", ["No (Open)", "Yes (Closed)"], horizontal=True)
    is_gate_closed = 1 if "Yes" in gate_input else 0
else:
    is_gate_closed = 0

weather_map = {"Clear / Sunny": 0, "Monsoon Rain / Waterlogging": 1, "Winter Dense Fog": 2}
selected_weather_label = st.selectbox("Weather Conditions:", list(weather_map.keys()))
selected_weather = weather_map[selected_weather_label]

st.markdown("---")

# ----------------- PREDICTION INFERENCE -----------------
if st.button("🚀 Predict Traffic Congestion"):
    input_data = np.array([[selected_corridor, hour, selected_festival, is_gate_closed, selected_weather]])
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    confidence = round(np.max(probs) * 100, 1)

    labels = ["🟢 Low / Free Flow", "🟡 Moderate Movement", "🟠 High Congestion", "🔴 Critical / Severe Gridlock"]
    
    st.markdown("### 📊 AI Prediction Result")
    
    if pred == 0:
        st.success(f"**Status:** {labels[pred]} *(Model Confidence: {confidence}%)*")
        st.info("💡 **Advisory:** Ideal travel window. No significant delays expected at ferry ghats or level crossings.")
    elif pred == 1:
        st.info(f"**Status:** {labels[pred]} *(Model Confidence: {confidence}%)*")
        st.write("💡 **Advisory:** Standard traffic flow. Minor bottlenecks possible near local market intersections.")
    elif pred == 2:
        st.warning(f"**Status:** {labels[pred]} *(Model Confidence: {confidence}%)*")
        st.write("💡 **Advisory:** Heavy crowd influx detected for this time! Expect 20–40 mins waiting time at Hulor Ghat/NH-12.")
    else:
        st.error(f"**Status:** {labels[pred]} *(Model Confidence: {confidence}%)*")
        st.write("🚨 **Critical Warning:** Peak festival crowd surge combined with corridor congestion! Consider deferring travel or using alternative rural bypass routes.")
