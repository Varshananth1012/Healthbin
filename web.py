from datetime import datetime
import streamlit as st
st.set_page_config(page_title="Health Monitor", layout="wide")

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

# Display the user profile
if st.session_state.user_profile:
    profile = st.session_state.user_profile
    st.markdown(f"""
        <div style="padding:10px; border-radius:10px; background-color:#f3f4f6; color:#111; margin-bottom:10px;">
        <h4>👤 User Profile</h4>
        <p><strong>Name:</strong> {profile['name']}</p>
        <p><strong>Age:</strong> {profile['age']}</p>
        <p><strong>Gender:</strong> {profile['gender']}</p>
        <p><strong>Email:</strong> {profile['email']}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Please create your profile in the sidebar before uploading a video.")


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.lite as tflite
import tempfile
from scipy.signal import detrend, butter, filtfilt

st.title("HEALTHBIN (BP, HR, SpO₂)")
st.caption("Your personal health monitor (BP, HR, SpO₂)")
st.write("Upload a fingertip video to analyze vital signs using PPG signals.")

# Upload video
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Preprocess function
def preprocess_ppg(ppg_signal, fs=30):
    ppg_signal = detrend(ppg_signal)
    b, a = butter(3, [0.7 / (0.5 * fs), 4 / (0.5 * fs)], btype='band')
    return filtfilt(b, a, ppg_signal)

if video_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps * 0.5)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        cap.release()

        if len(frames) < 1:
            st.error("No frames were extracted from the video.")
            st.stop()

        # Stack images
        stacked_image = np.mean(frames, axis=0).astype(np.uint8)
        st.image(stacked_image, caption="Stacked Image (RGB Average)")

        # Extract PPG signal
        r, g, b = cv2.split(stacked_image)
        ppg_signal = (np.mean(r, axis=1) + np.mean(g, axis=1) + np.mean(b, axis=1)) / 3
        ppg_signal = preprocess_ppg(ppg_signal)
        ppg_signal = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))
        time_axis = np.arange(len(ppg_signal))

        # Plot PPG
        st.subheader("📈 Extracted PPG Signal")
        fig, ax = plt.subplots()
        ax.plot(time_axis, ppg_signal, color="black")
        ax.set_xlabel("Time (rows)")
        ax.set_ylabel("Normalized Intensity")
        ax.set_title("PPG Signal")
        ax.grid()
        st.pyplot(fig)

        # Load TFLite model
        def run_model(tflite_path, ppg_signal, multi_output=False):
            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            expected_shape = input_details[0]['shape']
            resized = np.resize(ppg_signal, (expected_shape[1], 1)).astype(np.float32)
            resized = np.expand_dims(resized, axis=0)

            interpreter.set_tensor(input_details[0]['index'], resized)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            return output if multi_output else output[0]

        st.subheader("🧠 Model Predictions")

        bp_model = st.file_uploader("Upload BP model (.tflite)", type="tflite")
        hr_model = st.file_uploader("Upload HR model (.tflite)", type="tflite")
        spo2_model = st.file_uploader("Upload SpO₂ model (.tflite)", type="tflite")

        if bp_model and hr_model and spo2_model:
            with tempfile.NamedTemporaryFile(delete=False) as bp_tmp, \
                 tempfile.NamedTemporaryFile(delete=False) as hr_tmp, \
                 tempfile.NamedTemporaryFile(delete=False) as spo2_tmp:

                bp_tmp.write(bp_model.read())
                hr_tmp.write(hr_model.read())
                spo2_tmp.write(spo2_model.read())

                systolic, diastolic = run_model(bp_tmp.name, ppg_signal, multi_output=True)
                hr_value = run_model(hr_tmp.name, ppg_signal)
                spo2_value = run_model(spo2_tmp.name, ppg_signal)

                # Display results in colorful cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'''<div class="card"><h3>💧 Systolic</h3><p>{systolic:.2f} mmHg</p></div>''', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'''<div class="card"><h3>💧 Diastolic</h3><p>{diastolic:.2f} mmHg</p></div>''', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'''<div class="card"><h3>❤️ Heart Rate</h3><p>{hr_value:.2f} bpm</p></div>''', unsafe_allow_html=True)

                st.markdown(f'''<div class="card"><h3>🧪 SpO₂</h3><p>{spo2_value:.2f} %</p></div>''', unsafe_allow_html=True)
                

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #111827;
        color: white;
    }
    .card {
        background-color: #fef3c7;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        color: black;
    }
    .card h3 {
        font-weight: bold;
        color: black;
    }
    .card p {
        color: black;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
