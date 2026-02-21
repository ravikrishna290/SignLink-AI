import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==============================
# BASE MODEL DIRECTORY
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent / "models"


# ==============================
# LOAD MODEL BUNDLE
# ==============================
def load_bundle(folder):

    model = tf.keras.models.load_model(
        folder / "model.h5",
        compile=False   # 🚨 Prevent optimizer errors
    )

    scaler = joblib.load(folder / "scaler.pkl")
    labels = joblib.load(folder / "labels.pkl")

    model.trainable = False

    return model, scaler, labels


# ==============================
# LOAD ALL MODELS
# ==============================
basic7_model, basic7_scaler, basic7_labels = load_bundle(BASE_DIR / "basic7")
alphabet_model, alphabet_scaler, alphabet_labels = load_bundle(BASE_DIR / "alphabets")
numbers_model, numbers_scaler, numbers_labels = load_bundle(BASE_DIR / "numbers")
school_model, school_scaler, school_labels = load_bundle(BASE_DIR / "school")
public_model, public_scaler, public_labels = load_bundle(BASE_DIR / "public_places")
workplaces_model, workplaces_scaler, workplaces_labels = load_bundle(BASE_DIR / "workplaces")


# ==============================
# FEATURE CONVERTER & NORMALIZER
# ==============================
def to_126(features):
    features = np.array(features)

    if len(features) == 63:
        features = np.concatenate([features, np.zeros(63)])

    return features

def apply_scale_invariance(features_array):
    # Calculate the max absolute value in the feature array to find the bounding size
    max_val = np.max(np.abs(features_array))
    if max_val > 0:
        return features_array / max_val
    return features_array


# ==============================
# DECODE FUNCTION
# ==============================
def decode(pred, label_encoder):
    return label_encoder.inverse_transform([np.argmax(pred)])[0]


# ==============================
# MAIN PREDICT FUNCTION
# ==============================
def predict_all(features):

    outputs = {}

    features = np.array(features)

    # Ensure 126 version
    features_126 = to_126(features)
    
    # Apply Scale Invariance norm (Solves 50% accuracy!)
    scaled_126 = apply_scale_invariance(features_126)
    
    # Numbers uses 63 features
    features_63 = features[:63]
    scaled_63 = apply_scale_invariance(features_63)

    # ---- BASIC7 ----
    inp = basic7_scaler.transform(scaled_126.reshape(1, -1))
    pred = basic7_model.predict(inp, verbose=0)
    outputs["basic7"] = decode(pred, basic7_labels)

    # ---- ALPHABETS ----
    inp = alphabet_scaler.transform(scaled_126.reshape(1, -1))
    pred = alphabet_model.predict(inp, verbose=0)
    outputs["alphabets"] = decode(pred, alphabet_labels)

    # ---- NUMBERS (63 features) ----
    inp = numbers_scaler.transform(scaled_63.reshape(1, -1))
    pred = numbers_model.predict(inp, verbose=0)
    outputs["numbers"] = decode(pred, numbers_labels)

    # ---- SCHOOL ----
    inp = school_scaler.transform(scaled_126.reshape(1, -1))
    pred = school_model.predict(inp, verbose=0)
    outputs["school"] = decode(pred, school_labels)

    # ---- PUBLIC PLACES ----
    inp = public_scaler.transform(scaled_126.reshape(1, -1))
    pred = public_model.predict(inp, verbose=0)
    outputs["public_places"] = decode(pred, public_labels)

    # ---- WORKPLACES ----
    inp = workplaces_scaler.transform(scaled_126.reshape(1, -1))
    pred = workplaces_model.predict(inp, verbose=0)
    outputs["workplaces"] = decode(pred, workplaces_labels)

    return outputs
