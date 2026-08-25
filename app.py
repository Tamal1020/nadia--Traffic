import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="Global Traffic AI Predictor | Spatiotemporal Model",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional aesthetic formatting
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: 700; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6;}
    .alert-success {background-color: #D1FAE5; padding: 15px; border-radius: 5px; border-left: 5px solid #10B981;}
    .alert-warning {background-color: #FEF3C7; padding: 15px; border-radius: 5px; border-left: 5px solid #F59E0B;}
    .alert-danger {background-color: #FEE2E2; padding: 15px; border-radius: 5px; border-left: 5px solid #EF4444;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DICTIONARIES & MAPPINGS
# ==========================================
FESTIVAL_MAP = {
    "Normal Day (No Event)": 0,
    "Gaura Purnima (Feb-Mar)": 1,
    "Nityananda Trayodasi (Feb)": 2,
    "Snana Yatra (Jun)": 3,
    "Ratha Yatra (Jun-Jul)": 4,
    "Jhulan Yatra (Aug)": 5,
    "Sri Krishna Janmashtami (Aug-Sep)": 6,
    "Radhashtami (Sep)": 7,
    "Kartik Month / Damodara Vrata (Oct-Nov)": 8,
    "Gita Jayanti (Dec)": 9,
    "Durga Puja (Sep-Oct)": 10,
    "Jagaddhatri Puja (Nov)": 11,
    "Rash Yatra (Nov)": 12,
    "Saraswati Puja (Jan-Feb)": 13,
    "Eid-ul-Fitr / Eid-ul-Adha": 14,
    "Christmas / New Year (Dec-Jan)": 15
}

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

# ==========================================
# 3. MACHINE LEARNING ENGINE (BACKGROUND)
# ==========================================
@st.cache_resource
def build_and_train_model():
    """Generates hyper-realistic synthetic data and trains the Random Forest Model"""
    np.random.seed(42)
    num_samples = 4000

    # Generate synthetic features
    locations = np.random.choice([0, 1], size=num_samples) # 0: Bethuadahari, 1: Mayapur
    months = np.random.randint(1, 13, size=num_samples)
    hours = np.random.randint(0, 24, size=num_samples)
    days = np.random.randint(0, 7, size=num_samples)
    weather = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]) # 1: Rain/Flooded
    festivals = np.random.choice(list(FESTIVAL_MAP.values()), size=num_samples)
    
    # Simulate realistic railway gate closures based on Sealdah-Lalgola schedule
    train_passing = []
    for loc, hr in zip(locations, hours):
        if loc == 0:
            if hr in [2, 6, 7, 8, 9, 13, 15, 18, 19, 20, 22]:
                gate = np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                gate = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            gate = 0 # No railway at Mayapur Hulor Ghat
        train_passing.append(gate)

    # Compute deterministic congestion scores
    congestion_levels = []
    for loc, hr, fest, weath, gate, day in zip(locations, hours, festivals, weather, train_passing, days):
        score = 0.0
        
        # Temporal Impact (Diurnal Cycles)
        if hr in [8, 9, 10, 16, 17, 18, 19, 20]: score += 3.0
        elif hr in [11, 12, 13, 14, 15, 21]: score += 1.5
        else: score += 0.5
        
        # Weekend tourism penalty
        if day in [5, 6]: score += 1.5

        # Spatial & Cultural Rules
        if loc == 0: # Bethuadahari NH-12
            if gate == 1: score += 4.5
            if weath == 1: score += 2.0
            if fest in [10, 11]: score += 5.5
            if fest == 15 and (9 <= hr <= 17): score += 3.0
        else: # Mayapur Hulor Ghat
            if fest in [1, 4, 6]: score += 6.5
            if fest in [2, 3, 5, 7, 8, 12]: score += 3.5
            if weath == 1: score += 2.5
            if hr in [12, 13, 18, 19, 20]: score += 2.0
            
        # Target Classification
        if score >= 7.5: cong = 2 # Severe Gridlock
        elif score >= 4.0: cong = 1 # Moderate Delays
        else: cong = 0 # Clear
        congestion_levels.append(cong)

    df = pd.DataFrame({
        'Location': locations, 'Month': months, 'Hour': hours, 
        'Day_of_Week': days, 'Is_Festival': festivals, 
        'Train_Passing': train_passing, 'Weather': weather, 
        'Congestion': congestion_levels
    })

    X = df.drop('Congestion', axis=1)
    y = df['Congestion']
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X, y)
    return model

# Initialize Model
model = build_and_train_model()

# ==========================================
# 4. APP LAYOUT & USER INPUTS
# ==========================================
st.markdown('<div class="main-header">🌍 Global Traffic AI: Nadia District Micro-Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting complex traffic matrices integrating Cultural Festivals, Topology, and Transport Logistics.</div>', unsafe_allow_html=True)

st.sidebar.title("Configuration Panel")
st.sidebar.markdown("Enter your travel parameters below to generate a hyper-realistic prediction.")

# Sidebar Inputs
route_input = st.sidebar.selectbox("Select Traffic Corridor:", ["Bethuadahari (NH-12 Level Crossing)", "Mayapur (Hulor Ghat Road)"])
month_input = st.sidebar.selectbox("Select Month:", list(MONTH_MAP.keys()))
day_input = st.sidebar.selectbox("Select Day of Week:", list(DAY_MAP.keys()))
hour_input = st.sidebar.slider("Hour of the Day (24H Format):", 0, 23, 14)
festival_input = st.sidebar.selectbox("Cultural Event / Festival:", list(FESTIVAL_MAP.keys()))
weather_input = st.sidebar.radio("Weather Condition:", ["Clear / Normal", "Heavy Rain / Monsoon"])
gate_input = st.sidebar.radio("Railway Gate Status (Bethuadahari Only):", ["Open / Unknown", "Currently Closed"])

# ==========================================
# 5. PREDICTION LOGIC & INFERENCE
# ==========================================
if st.sidebar.button("🚀 Execute AI Prediction", use_container_width=True):
    
    loc_val = 0 if "Bethuadahari" in route_input else 1
    month_val = MONTH_MAP[month_input]
    day_val = DAY_MAP[day_input]
    fest_val = FESTIVAL_MAP[festival_input]
    weath_val = 1 if "Rain" in weather_input else 0
    gate_val = 1 if ("Closed" in gate_input and loc_val == 0) else 0

    input_data = pd.DataFrame([[loc_val, month_val, hour_input, day_val, fest_val, gate_val, weath_val]], 
                              columns=['Location', 'Month', 'Hour', 'Day_of_Week', 'Is_Festival', 'Train_Passing', 'Weather'])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # ==========================================
    # 6. RESULTS & INSIGHTS DASHBOARD
    # ==========================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    col1.markdown(f"<div class='metric-card'><strong>📍 Corridor:</strong><br>{route_input}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><strong>⏰ Time:</strong><br>{hour_input}:00, {day_input}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><strong>🎭 Cultural Context:</strong><br>{festival_input}</div>", unsafe_allow_html=True)
    
    st.markdown("### 🚦 AI Traffic Verdict")
    
    if prediction == 0:
        st.markdown("""<div class='alert-success'>
        <strong>🟢 LOW CONGESTION (Clear Route)</strong><br>
        The corridor is experiencing baseline traffic. You can proceed with minimal delays.
        </div>""", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("""<div class='alert-warning'>
        <strong>🟡 MODERATE CONGESTION (Expect Delays)</strong><br>
        Traffic is building up due to time-of-day dynamics or localized events. Buffer your travel time by 20-30 minutes.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='alert-danger'>
        <strong>🔴 SEVERE GRIDLOCK (Avoid Route)</strong><br>
        The corridor is entirely choked. This is a compounding effect of festival crowds, topographical limits (rail gates/ferries), and peak hours.
        </div>""", unsafe_allow_html=True)

    # ==========================================
    # 7. VEHICLE VOLUME & RECOMMENDATION ENGINE
    # ==========================================
    st.markdown("### 🚙 Vehicle Volume Analysis & Modal Recommendations")
    
    if loc_val == 1: # Mayapur
        if fest_val in [1, 4, 6, 8, 12]:
            highest_vol = "Pedestrians & Devotees (Millions)"
            lowest_vol = "Four-Wheelers (Physically restricted)"
            rec = "Motorized transit is paralyzed. *Walking* is the only viable method near Hulor Ghat. Ferries may have extreme queues."
        elif hour_input in range(10, 16):
            highest_vol = "E-Rickshaws (Totos) & Auto-Rickshaws"
            lowest_vol = "Heavy Commercial Vehicles"
            rec = "*Two-wheelers* are optimal for slicing through Toto gridlocks near the temple gates."
        else:
            highest_vol = "Mixed Two/Four Wheelers"
            lowest_vol = "Commercial Transport"
            rec = "Standard vehicles are viable. If raining heavily, avoid the ferry ghat approach roads as they become impassable."
            
    else: # Bethuadahari
        if fest_val in [10, 11]: # Durga / Jagaddhatri
            highest_vol = "Pedestrians, Pandals, & Local Totos"
            lowest_vol = "Highway Transit (Halted)"
            rec = "NH-12 acts as a parking lot during peak puja hours. Bypass the town entirely or rely on *Two-wheelers* via village inner roads."
        elif gate_val == 1 or hour_input in [6, 7, 8, 9, 18, 19, 20]:
            highest_vol = "Multi-Axle Trucks & Intercity Buses (Idling)"
            lowest_vol = "Pedestrians"
            rec = "Lalgola train passage creates massive highway queues. *Four-wheelers* will be trapped. Plan transit outside these windows."
        else:
            highest_vol = "Heavy Commercial Vehicles"
            lowest_vol = "Local Totos"
            rec = "The highway is flowing. *Cars and SUVs* are the safest and most efficient mode right now."

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.info(f"📈 **Dominant Traffic Profile:**\n\n{highest_vol}")
        st.error(f"📉 **Lowest Usable Volume:**\n\n{lowest_vol}")
    with r_col2:
        st.success(f"💡 **AI Recommendation:**\n\n{rec}")

    st.caption(f"Model Confidence: {max(probabilities)*100:.1f}% based on Random Forest ensemble logic.")
