# F1-tire-retention
F1 tire retention and pit stop strategy



# F1 Tyre Retention & Pit‑Stop Strategy – ML + Streamlit

This project builds a machine learning model and Streamlit app to predict **when a Formula 1 car should pit** and to compare simple **“stay out vs pit now”** strategies over the next few laps.

## 1. Project overview

- **Goal:** For each lap, estimate whether a car should **pit within the next N laps** or continue, and show a basic time comparison of staying out vs pitting now. 
- **Type:** Binary classification + simple strategy simulation.  
- **Tech:** Python, pandas, scikit‑learn, Streamlit. [web:119][web:136]

**Key components**

- Data processing from multiple F1 tables (laps, telemetry, event info, results). 
- Random Forest classifier to predict `pit_soon` (0/1).  
- Streamlit web UI for interactive predictions and strategy comparison. 

## 2. Data and features

The model is trained on lap‑level rows: one row = one driver on one lap.

**Main feature groups**

- **Lap & timing**  
  - `LapNumber`, `LapTime`, `Sector1Time`, `Sector2Time`, `Sector3Time` (converted to seconds).
- **Tyre / stint**  
  - `Stint`, `TyreLife` (laps on current set), `FreshTyre` (bool), `Compound` (Soft/Medium/Hard). 
- **Race context**  
  - `Position`, `GridPosition`, `TrackStatus` (green, SC, etc.). 
- **Telemetry summaries**  
  - `Speed_mean`, `Speed_max`, `Throttle_mean`, `Brake_mean`.
- **Categorical one‑hots**  
  - `Compound_*`, `TrackStatus_*`, `Driver_*`. 

**Target label**

- `pit_soon` = 1 if a pit stop occurs within the next **N laps** after this lap; otherwise 0. 
- Pit‑timing columns and final results are **excluded** from X to avoid leakage.

## 3. Modeling

- **Model:** `RandomForestClassifier` with `class_weight='balanced'` to handle rare pit‑soon laps. 
- **Training:**  
  - Build final encoded feature matrix `X` and label `y`.  
  - Train/test split.  
  - Save model (`pit_model.pkl`) and feature column list (`feature_columns.pkl`). 

**Metrics (example)**

- Accuracy ≈ **0.99**.  
- For `pit_soon = 1`: precision ≈ **0.85**, recall ≈ **0.96**, F1 ≈ **0.91**.  

## 4. Streamlit app

### 4.1. Inputs

Sidebar controls for the current lap state:

- Lap inputs: Lap number, lap time, sector times, stint number, tyre life, current position, grid position.
- Tyres & context: Tyre compound (Soft/Medium/Hard), fresh tyre flag, track status code, driver. 
- Telemetry: Mean speed, max speed, mean throttle, mean brake.

### 4.2. Outputs

1. **Pit‑stop prediction**

- `Pit in next N laps (probability)` – probability that `pit_soon = 1`.  
- `Pit in next N laps (class)` – 0/1 based on a decision threshold (default 0.5, can be made configurable).
2. **Simple strategy comparison**

- Inputs: horizon (laps to look ahead), pit‑lane time loss (s), degradation per lap (s/lap), fresh‑tyre gain (s). 
- Computes total time if you **stay out** vs **pit now** over the horizon, then indicates which is faster. 

## 5. How to run

```bash
pip install -r requirements.txt
# or at minimum:
pip install streamlit scikit-learn pandas numpy

python -m streamlit run app.py #run in the terminal
