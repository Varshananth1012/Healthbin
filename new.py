from datetime import datetime
import streamlit as st
import os
import numpy as np
import tempfile
import cv2
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
import tensorflow.lite as tflite
import sqlite3
import pandas as pd
import requests

# App Config
st.set_page_config(page_title="OxyPulse", layout="wide")

# DB Setup
conn = sqlite3.connect("oxypulse.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        date TEXT,
        systolic REAL,
        diastolic REAL,
        heart_rate REAL,
        spo2 REAL
    )
''')
conn.commit()

# Sidebar Profile
st.sidebar.header("👤 Create Your Profile")
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

with st.sidebar.form("profile_form"):
    name = st.text_input("Name", "")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    email = st.text_input("Email", "")
    submitted = st.form_submit_button("Save Profile")

    if submitted:
        st.session_state.user_profile = {
            "name": name,
            "age": age,
            "gender": gender,
            "email": email
        }
        st.success(f"Profile saved for {name}!")

# Show profile
if st.session_state.user_profile:
    profile = st.session_state.user_profile
    st.markdown(f"""
        <div style="padding:10px; border-radius:10px; background:linear-gradient(to right, #fff1eb, #ace0f9); color:#000;">
        <h4>👤 User Profile</h4>
        <p><strong>Name:</strong> {profile['name']}</p>
        <p><strong>Age:</strong> {profile['age']}</p>
        <p><strong>Gender:</strong> {profile['gender']}</p>
        <p><strong>Email:</strong> {profile['email']}</p>
        </div>
    """, unsafe_allow_html=True)

# Upload
st.title("OxyPulse (BP, HR, SpO₂)")
video_file = st.file_uploader("Upload a fingertip video", type=["mp4", "avi", "mov"])

def preprocess_ppg(ppg_signal, fs=30):
    ppg_signal = detrend(ppg_signal)
    b, a = butter(3, [0.7 / (0.5 * fs), 4 / (0.5 * fs)], btype='band')
    return filtfilt(b, a, ppg_signal)

def run_model(tflite_path, ppg_signal, multi_output=False):
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    resized = np.resize(ppg_signal, (input_details[0]['shape'][1], 1)).astype(np.float32)
    resized = np.expand_dims(resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], resized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output if multi_output else output[0]

def download_model_if_not_exists(url, local_path):
    if not os.path.exists(local_path):
        st.info(f"Downloading model from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            st.success(f"Model saved to {local_path}")
        else:
            st.error(f"Failed to download model from {url}")

# Main
if video_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps * 0.5)
        frames, frame_count = [], 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        cap.release()

        if not frames:
            st.error("No frames extracted.")
            st.stop()

        stacked_image = np.mean(frames, axis=0).astype(np.uint8)
        st.image(stacked_image, caption="Stacked Image")

        r, g, b = cv2.split(stacked_image)
        ppg_signal = (np.mean(r, axis=1) + np.mean(g, axis=1) + np.mean(b, axis=1)) / 3
        ppg_signal = preprocess_ppg(ppg_signal)
        ppg_signal = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(ppg_signal)), ppg_signal, color="black")
        ax.set_title("Extracted PPG Signal")
        ax.grid()
        st.pyplot(fig)

        # Model paths
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        # GitHub raw links (Replace with your actual raw links)
        bp_url = "https://raw.githubusercontent.com/Varshananth1012/Healthbin/main/models/bp_model.tflite"
        hr_url = "https://raw.githubusercontent.com/Varshananth1012/Healthbin/main/models/hr_model.tflite"
        spo2_url = "https://raw.githubusercontent.com/Varshananth1012/Healthbin/main/models/spo2_model.tflite"

        # Local paths
        bp_model_path = os.path.join(model_dir, "bp_model.tflite")
        hr_model_path = os.path.join(model_dir, "hr_model.tflite")
        spo2_model_path = os.path.join(model_dir, "spo2_model.tflite")

        # Download models
        download_model_if_not_exists(bp_url, bp_model_path)
        download_model_if_not_exists(hr_url, hr_model_path)
        download_model_if_not_exists(spo2_url, spo2_model_path)

        # Predictions
        if os.path.exists(bp_model_path) and os.path.exists(hr_model_path) and os.path.exists(spo2_model_path):
            systolic, diastolic = run_model(bp_model_path, ppg_signal, multi_output=True)
            hr_value = run_model(hr_model_path, ppg_signal)
            spo2_value = run_model(spo2_model_path, ppg_signal)

            st.subheader("🧠 Model Predictions")
            st.write(f"💧 **Systolic:** {systolic:.2f} mmHg")
            st.write(f"💧 **Diastolic:** {diastolic:.2f} mmHg")
            st.write(f"❤️ **Heart Rate:** {hr_value:.2f} bpm")
            st.write(f"🧪 **SpO₂:** {spo2_value:.2f} %")

            # Save to DB
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO predictions (name, email, date, systolic, diastolic, heart_rate, spo2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (profile['name'], profile['email'], now, systolic, diastolic, hr_value, spo2_value))
            conn.commit()
        else:
            st.error("Model files not found.")

# Show History
if st.session_state.user_profile:
    st.subheader("📊 Prediction History")
    user_email = st.session_state.user_profile['email']
    cursor.execute("SELECT date, systolic, diastolic, heart_rate, spo2 FROM predictions WHERE email = ? ORDER BY date DESC", (user_email,))
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=["Date", "Systolic", "Diastolic", "Heart Rate", "SpO₂"])
    st.dataframe(df)

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Report", data=csv, file_name="oxypulse_predictions.csv", mime="text/csv")

# Style
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        color: black;
    }
    .css-1cpxqw2 {
        background-color: white;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)