import datetime
import joblib
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Wildfire Risk Predictor",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Wildfire Risk Prediction System")
st.write(
    "Enter a location's latitude and longitude to fetch the last 7 days of atmospheric weather data "
    "and calculate the wildfire risk using Machine Learning."
)

# Load trained model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('fire_model.pkl')
    scaler_obj = joblib.load('scaler.pkl')
    return model, scaler_obj

try:
    lr_model, scaler = load_assets()
except Exception as e:
    st.error("⚠️ Model files (`fire_model.pkl` or `scaler.pkl`) not found in the root directory!")
    st.info("Please make sure both `.pkl` files are placed in the same folder as `app.py`.")
    st.stop()

# User Inputs for Coordinates
st.subheader("📍 Input Location Coordinates")
col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Latitude", value=26.4499, format="%.4f")
with col2:
    lon = st.number_input("Longitude", value=80.3319, format="%.4f")

if st.button("Assess Wildfire Risk", type="primary"):
    with st.spinner("Fetching past 7 days of hourly weather data from Open-Meteo API..."):
        # Fetch past 7 days of hourly data via Open-Meteo
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"past_days=7&forecast_days=0&"
            f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&"
            f"timezone=auto"
        )
        
        response = requests.get(url)

    if response.status_code == 200:
        hourly = response.json().get('hourly', {})
        
        if hourly and 'time' in hourly:
            # Create hourly dataframe
            df_hourly = pd.DataFrame({
                'time': pd.to_datetime(hourly['time']),
                'temperature': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'wind_speed': hourly['wind_speed_10m'],
                'precipitation': hourly['precipitation']
            }).dropna()

            # Group by date: mean for temp/humidity/wind, sum for rainfall
            df_hourly['date'] = df_hourly['time'].dt.date
            df_daily = df_hourly.groupby('date').agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'wind_speed': 'mean',
                'precipitation': 'sum'
            })

            # Calculate 7-day averages
            avg_weather = df_daily.mean().to_frame().T.round(2)
            feature_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']

            # Feature Scaling & Probability Prediction
            scaled_input = scaler.transform(avg_weather[feature_cols])
            risk_probability = lr_model.predict_proba(scaled_input)[0][1] * 100

            # Display 7-day Averages Table
            st.subheader("📊 Past 7-Day Average Weather Features")
            st.dataframe(
                avg_weather.rename(columns={
                    'temperature': 'Avg Temp (°C)',
                    'humidity': 'Avg Humidity (%)',
                    'wind_speed': 'Avg Wind Speed (km/h)',
                    'precipitation': 'Avg Daily Rain (mm)'
                }),
                use_container_width=True
            )

            # Display Risk Outcome
            st.subheader("🔥 Fire Risk Evaluation")
            
            if risk_probability >= 65.0:
                st.error(
                    f"🚨 **HIGH FIRE RISK ({risk_probability:.1f}% Chance)**\n\n"
                    "High probability of fire outbreak. High temperatures, strong winds, or low atmospheric humidity detected."
                )
            elif 30.0 <= risk_probability < 65.0:
                st.warning(
                    f"⚠️ **MODERATE RISK ({risk_probability:.1f}% Chance)**\n\n"
                    "Caution: There are **some chances of fire**. Atmospheric conditions indicate moderate risk."
                )
            else:
                st.success(
                    f"✅ **LOW RISK ({risk_probability:.1f}% Chance)**\n\n"
                    "Safe weather conditions. Low risk of fire based on recent 7-day atmospheric averages."
                )

        else:
            st.warning("No valid weather data returned for these coordinates.")
    else:
        st.error(f"Failed to retrieve weather data from Open-Meteo. HTTP Error Code: {response.status_code}")