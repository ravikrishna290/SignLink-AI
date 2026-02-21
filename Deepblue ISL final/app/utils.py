import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# FRAME LANDMARK EXTRACTION
# ==============================
def extract_landmarks(frame):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    annotated_frame = frame.copy()

    if results.multi_hand_landmarks:

        landmarks = []

        for hand_landmarks in results.multi_hand_landmarks[:2]:

            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Force 126 features
        if len(landmarks) < 126:
            landmarks = np.concatenate([landmarks, np.zeros(126 - len(landmarks))])
        elif len(landmarks) > 126:
            landmarks = landmarks[:126]

        return landmarks, annotated_frame

    return None, annotated_frame


# ==============================
# VIDEO LANDMARK EXTRACTION
# ==============================
def extract_from_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    all_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, _ = extract_landmarks(frame)

        if landmarks is not None:
            all_landmarks.append(landmarks)

    cap.release()

    if len(all_landmarks) == 0:
        return None

    # Average landmarks across frames
    return np.mean(all_landmarks, axis=0)
