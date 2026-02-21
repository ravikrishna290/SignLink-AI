import numpy as np

# ==============================
# FEATURE SIZE HANDLING
# ==============================

def to_126(features):
    """
    Convert 63 → 126 if needed
    """
    features = np.array(features)

    if features.shape[0] == 63:
        return np.concatenate([features, np.zeros(63)])

    return features


def to_63(features):
    """
    Extract first 63 features
    (Used for numbers model)
    """
    features = np.array(features)

    if features.shape[0] >= 63:
        return features[:63]

    raise ValueError("Input features < 63")


# ==============================
# RESHAPING
# ==============================

def reshape_for_model(features):
    """
    Convert → (1, N)
    """
    return np.array(features).reshape(1, -1)


# ==============================
# SCALING
# ==============================

def scale_features(features, scaler):
    """
    Apply StandardScaler safely
    """
    return scaler.transform(reshape_for_model(features))


# ==============================
# MODEL-SPECIFIC PREPROCESSORS
# ==============================

def preprocess_126(features, scaler):
    """
    For models expecting 126 features
    """
    features_126 = to_126(features)
    return scale_features(features_126, scaler)


def preprocess_63(features, scaler):
    """
    For numbers model (63 features)
    """
    features_63 = to_63(features)
    return scale_features(features_63, scaler)
