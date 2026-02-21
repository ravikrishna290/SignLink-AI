import cv2
import streamlit as st
import numpy as np
from pathlib import Path

from app.predictor import predict_all
from app.utils import extract_landmarks

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="ISL Live Detection", layout="wide")

st.title("🤟 ISL Live Sign Detection")
st.write("Live camera → MediaPipe → Multi-model prediction")

# ==============================
# CAMERA START / STOP
# ==============================
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
PREDICTION_BOX = st.empty()

# ==============================
# VIDEO CAPTURE
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access webcam")
    st.stop()

while run:

    ret, frame = cap.read()

    if not ret:
        st.warning("⚠ Frame not captured")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # ==============================
    # LANDMARK EXTRACTION
    # ==============================
    landmarks, annotated_frame = extract_landmarks(frame)

    if landmarks is not None:

        # ==============================
        # PREDICTION
        # ==============================
        results = predict_all(landmarks)

        PREDICTION_BOX.markdown(
            f"""
            ### 🔮 Predictions
            **Basic7:** {results['basic7']}  
            **Alphabets:** {results['alphabets']}  
            **Numbers:** {results['numbers']}  
            **School:** {results['school']}  
            **Public Places:** {results['public_places']}  
            **Workplaces:** {results['workplaces']}  
            """
        )

    else:
        PREDICTION_BOX.markdown("### ✋ No hand detected")

    # Convert BGR → RGB
    FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()
