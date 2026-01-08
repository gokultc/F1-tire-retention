# app.py
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Load model and feature columns ----------
@st.cache_resource
def load_model():
    with open("pit_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols


model, FEATURE_COLS = load_model()


# ---------- Helper: build one-row input ----------
def build_input_df(
    lap_number,
    lap_time_sec,
    s1_sec,
    s2_sec,
    s3_sec,
    stint,
    tyre_life,
    fresh_tyre,
    position,
    speed_mean,
    speed_max,
    throttle_mean,
    brake_mean,
    grid_position,
    compound,
    track_status,
    driver
):
    # base dict with all raw fields
    data = {
        "LapNumber": [lap_number],
        "LapTime": [lap_time_sec],
        "Sector1Time": [s1_sec],
        "Sector2Time": [s2_sec],
        "Sector3Time": [s3_sec],
        "Stint": [stint],
        "TyreLife": [tyre_life],
        "FreshTyre": [fresh_tyre],
        "Position": [position],
        "Speed_mean": [speed_mean],
        "Speed_max": [speed_max],
        "Throttle_mean": [throttle_mean],
        "Brake_mean": [brake_mean],
        "GridPosition": [grid_position],
        "Compound": [compound],
        "TrackStatus": [track_status],
        "Driver": [driver],
    }

    df = pd.DataFrame(data)

    # apply same get_dummies as in training
    df_enc = pd.get_dummies(
        df,
        columns=["Compound", "TrackStatus", "Driver"],
        drop_first=True,
    )

    # ensure all expected columns exist and in correct order
    for col in FEATURE_COLS:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc = df_enc[FEATURE_COLS]

    return df_enc


def eval_pit_strategy(
    current_lap_time,
    current_tyre_life,
    horizon_laps=5,
    pit_lane_loss=20.0,
    deg_per_lap=0.15,
    fresh_tyre_gain=1.0
):
    """
    Simple time comparison:
    - Stay out: lap time increases with tyre age.
    - Pit now: pay pit lane loss, then run laps on fresher tyres.
    """
    # Scenario B: stay out
    times_stay = []
    for h in range(horizon_laps):
        lap_time = current_lap_time + deg_per_lap * (current_tyre_life + h)
        times_stay.append(lap_time)
    total_stay = sum(times_stay)

    # Scenario A: pit now
    base_fresh = max(0.0, current_lap_time - fresh_tyre_gain)
    times_pit = [base_fresh for _ in range(horizon_laps)]
    total_pit = pit_lane_loss + sum(times_pit)

    return total_stay, total_pit


# ---------- Streamlit UI ----------
st.title("F1 Pit-Stop Prediction (Pit in Next N Laps)")

st.sidebar.header("Lap Inputs")


# Basic numeric inputs
lap_number = st.sidebar.number_input("Lap Number", min_value=1, value=10)
lap_time_sec = st.sidebar.number_input("Lap Time (seconds)", min_value=0.0, value=90.0)
s1_sec = st.sidebar.number_input("Sector 1 Time (s)", min_value=0.0, value=30.0)
s2_sec = st.sidebar.number_input("Sector 2 Time (s)", min_value=0.0, value=30.0)
s3_sec = st.sidebar.number_input("Sector 3 Time (s)", min_value=0.0, value=30.0)
stint = st.sidebar.number_input("Stint Number", min_value=1, value=1)
tyre_life = st.sidebar.number_input("Tyre Life (laps)", min_value=0, value=5)
fresh_tyre = st.sidebar.checkbox("Fresh Tyre", value=False)
position = st.sidebar.number_input("Current Position", min_value=1, value=5)
speed_mean = st.sidebar.number_input("Mean Speed (km/h)", min_value=0.0, value=220.0)
speed_max = st.sidebar.number_input("Max Speed (km/h)", min_value=0.0, value=330.0)
throttle_mean = st.sidebar.number_input("Mean Throttle (%)", min_value=0.0, max_value=100.0, value=80.0)
brake_mean = st.sidebar.number_input("Mean Brake (%)", min_value=0.0, max_value=100.0, value=20.0)
grid_position = st.sidebar.number_input("Grid Position", min_value=1, value=5)

# Categorical inputs
compound = st.sidebar.selectbox("Tyre Compound", ["HARD", "MEDIUM", "SOFT"])
track_status = st.sidebar.selectbox("Track Status Code", ["1", "2", "12", "21", "126", "671", "2671"])
driver = st.sidebar.selectbox(
    "Driver",
    ["ALO","BOT","DEV","GAS","HAM","HUL","LEC","MAG",
     "NOR","OCO","PER","PIA","RUS","SAI","SAR",
     "STR","TSU","VER","ZHO"]
)

threshold = st.slider(
    "Decision threshold for pit soon",
    0.0, 1.0, 0.5
)

# ----- ML prediction -----
if st.button("Predict Pit Soon"):
    input_df = build_input_df(
        lap_number,
        lap_time_sec,
        s1_sec,
        s2_sec,
        s3_sec,
        stint,
        tyre_life,
        fresh_tyre,
        position,
        speed_mean,
        speed_max,
        throttle_mean,
        brake_mean,
        grid_position,
        compound,
        track_status,
        driver,
    )

    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob > threshold)   # use custom threshold

    st.subheader("Prediction")
    st.write(f"Pit in next N laps (class): **{pred}**")
    st.write(f"Pit in next N laps (probability): **{prob:.3f}**")


# ----- Simple strategy comparison -----
st.subheader("Simple Strategy Comparison (Stay vs Pit Now)")

horizon = st.slider("Horizon (laps to look ahead)", 3, 15, 5)
pit_loss = st.number_input("Pit lane time loss (s)", min_value=10.0, max_value=30.0, value=20.0)
deg = st.number_input("Degradation per lap (s/lap)", min_value=0.0, max_value=1.0, value=0.15)
fresh_gain = st.number_input("Fresh tyre gain vs now (s)", min_value=0.0, max_value=3.0, value=1.0)

if st.button("Evaluate Strategy"):
    total_stay, total_pit = eval_pit_strategy(
        current_lap_time=lap_time_sec,
        current_tyre_life=tyre_life,
        horizon_laps=horizon,
        pit_lane_loss=pit_loss,
        deg_per_lap=deg,
        fresh_tyre_gain=fresh_gain
    )

    st.write(f"Total time if you **stay out** (next {horizon} laps): **{total_stay:.2f} s**")
    st.write(f"Total time if you **pit now** (next {horizon} laps): **{total_pit:.2f} s**")

    if total_pit < total_stay:
        st.success("Pitting now is faster over this horizon.")
    else:
        st.info("Staying out is faster over this horizon.")
