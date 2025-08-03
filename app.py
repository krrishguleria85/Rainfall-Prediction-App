import streamlit as st
import pandas as pd
import joblib
import os
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rainfall Predictor",
    page_icon="🌦️",
    layout="centered"
)

# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_model_and_features():
    """Load the saved model and the list of features it was trained on."""
    model_path = 'weather_prediction_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"Fatal Error: Model file '{model_path}' not found.")
        st.info("Please make sure the model is trained and saved in the same directory.")
        st.stop()
    
    try:
        data = joblib.load(model_path)
        return data['model'], data['features']
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        st.stop()

model, model_features = load_model_and_features()

# --- HELPER & API FUNCTIONS ---
def degrees_to_cardinal(d):
    """Convert wind direction in degrees to a cardinal direction (e.g., N, E, SW)."""
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = int(round(d / (360. / len(dirs))))
    return dirs[ix % len(dirs)]

@st.cache_data(ttl=600) # Cache the weather data for 10 minutes
def fetch_weather_data(city, api_key):
    """Fetch accurate weather data from OpenWeatherMap API."""
    if not api_key:
        return None, "API Key is not configured. Please add it to your secrets.toml file."
    if not city:
        return None, "City name is missing."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_details = {
            'MinTemp': data['main']['temp_min'],
            'MaxTemp': data['main']['temp_max'],
            'WindGustSpeed': data.get('wind', {}).get('gust', data.get('wind', {}).get('speed', 0)) * 3.6, # m/s to km/h
            'Humidity': data['main']['humidity'],
            'Pressure': data['main']['pressure'],
            'Temp': data['main']['temp'],
            'WindGustDir': degrees_to_cardinal(data.get('wind', {}).get('deg', 0))
        }
        return weather_details, None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return None, "Invalid API Key in secrets.toml. Please check your key or wait for it to become active."
        elif e.response.status_code == 404:
            return None, f"City '{city}' not found. Please check the spelling."
        else:
            return None, f"An HTTP error occurred: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

# --- UI STYLING ---
st.markdown("""
<style>
    /* Main App Styling */
    .stApp {
        background-color: #0066ff; 
    }
    /* Main Title */
    h1 {
        color: #000099;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    /* Subheaders */
    h2, h3 {
        color: #000099;
    }
    /* Result Card */
    .result-card {
        background: #000099; 
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        border: 2px solid;
    }
    .result-yes { border-color: #000099; }
    .result-no { border-color: #000099; }
    .result-icon { font-size: 4rem; }
    .result-text { font-size: 2.5rem; font-weight: bold; color: #000099; }
</style>
""", unsafe_allow_html=True)


# --- APP LAYOUT ---
st.title("Weather Prediction (Rainfall) 🌧️")
st.markdown("---")

# --- USER INPUT ON MAIN PAGE ---
st.header("Enter Location")
city = st.text_input("City Name", "Chandigarh", label_visibility="collapsed")

# --- LOAD API KEY FROM SECRETS ---
try:
    api_key = st.secrets["OPENWEATHER_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("API Key not found. Please ensure you have a .streamlit/secrets.toml file with your key.")
    st.stop()

# --- MAIN CONTENT ---
if city:
    weather_data, error_message = fetch_weather_data(city, api_key)
    
    if error_message:
        st.error(error_message)
    elif weather_data:
        st.markdown("---")
        st.header(f"Live Weather in {city.title()}")
        
        # Display current weather in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather_data['Temp']} °C")
        col2.metric("Humidity", f"{weather_data['Humidity']}%")
        col3.metric("Pressure", f"{weather_data['Pressure']} hPa")

        # Prediction button and logic
        if st.button("Predict Tomorrow's Rainfall", use_container_width=True, type="primary"):
            with st.spinner("Analyzing weather patterns..."):
                # Prepare data for model
                input_data = weather_data.copy()
                wind_dir = input_data.pop('WindGustDir')
                
                for feature in model_features:
                    if 'WindGustDir_' in feature:
                        input_data[feature] = 1 if feature.endswith(wind_dir) else 0
                
                input_df = pd.DataFrame([input_data])[model_features]
                
                # Predict
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                # Display result
                if prediction == 1:
                    st.markdown('<div class="result-card result-yes">', unsafe_allow_html=True)
                    st.markdown('<p class="result-icon">☔</p>', unsafe_allow_html=True)
                    st.markdown('<p class="result-text">Yes, rain is likely tomorrow.</p>', unsafe_allow_html=True)
                    st.metric("Probability of Rain", f"{probability*100:.2f}%")
                else:
                    st.markdown('<div class="result-card result-no">', unsafe_allow_html=True)
                    st.markdown('<p class="result-icon">☀️</p>', unsafe_allow_html=True)
                    st.markdown('<p class="result-text">No, rain is not expected tomorrow.</p>', unsafe_allow_html=True)
                    st.metric("Probability of Rain", f"{probability*100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please enter a City Name above to start.")
