import streamlit as st
import joblib
import numpy as np

model = joblib.load('pkl/tornado_model.pkl')
state_le = joblib.load('pkl/state_encoder.pkl')
month_le = joblib.load('pkl/month_encoder.pkl')
scale_le = joblib.load('pkl/scale_encoder.pkl')

st.title("Tornado Severity Predictor")
st.write("Enter storm details to predict tornado severity (EF0-EF5)")

month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"][x-1])

state = st.selectbox("State", sorted(state_le.classes_))

tor_length = st.number_input("Tornado Length (miles)", min_value=0.0, value=1.0)
tor_width = st.number_input("Tornado Width (yards)", min_value=0.0, value=50.0)
injuries = st.number_input("Injuries", min_value=0, value=0)
deaths = st.number_input("Deaths", min_value=0, value=0)
damage = st.number_input("Property Damage ($)", min_value=0.0, value=0.0)
begin_lat = st.number_input("Latitude", value=35.0)
begin_lon = st.number_input("Longitude", value=-97.0)

if st.button("Predict Severity"):
    month_name = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"][month-1]

    month_encoded = month_le.transform([month_name])[0]
    state_encoded = state_le.transform([state])[0]

    features = np.array([[month_encoded, state_encoded, tor_length, tor_width,
                          injuries, deaths, damage, begin_lat, begin_lon]])
    
    prediction = model.predict(features)[0]
    severity = scale_le.inverse_transform([prediction])[0]

    st.success(f"Predicted Tornado Severity: **{severity}**")
