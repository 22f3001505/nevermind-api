"""
Career Path AI — ML Inference Module
Loads the trained RandomForest model and provides prediction functions.
"""
import os
import pickle
import numpy as np


# --- Load model artifacts at module level (singleton pattern) ---
_MODEL_DIR = os.path.dirname(__file__)
_model = None
_label_encoder = None
_meta = None


def _load_artifacts():
    """Lazy-load model artifacts on first use."""
    global _model, _label_encoder, _meta

    if _model is not None:
        return

    model_path = os.path.join(_MODEL_DIR, "career_model.pkl")
    encoder_path = os.path.join(_MODEL_DIR, "label_encoder.pkl")
    meta_path = os.path.join(_MODEL_DIR, "model_meta.pkl")

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        _label_encoder = pickle.load(f)

    with open(meta_path, "rb") as f:
        _meta = pickle.load(f)


def get_skill_columns():
    """Return the ordered list of skill columns the model expects."""
    _load_artifacts()
    return _meta["skill_columns"]


def get_career_list():
    """Return all possible career labels."""
    _load_artifacts()
    return _meta["careers"]


def predict_careers(skill_vector, top_n=3):
    """
    Predict career paths from a skill vector.

    Args:
        skill_vector: dict mapping skill names to float scores (0.0-1.0)
                      e.g. {"python": 0.8, "javascript": 0.3, ...}
        top_n: Number of top career predictions to return (default: 3)

    Returns:
        list of dicts: [{"career": str, "confidence": float, "rank": int}, ...]
    """
    _load_artifacts()

    # Build feature array in correct column order
    columns = _meta["skill_columns"]
    features = np.array([skill_vector.get(col, 0.0) for col in columns]).reshape(1, -1)

    # Get probabilities for all classes
    probabilities = _model.predict_proba(features)[0]

    # Sort by probability (descending)
    class_indices = np.argsort(probabilities)[::-1]

    results = []
    for rank, idx in enumerate(class_indices[:top_n], start=1):
        career_name = _label_encoder.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append({
            "career": career_name,
            "confidence": round(confidence, 4),
            "rank": rank
        })

    return results


def predict_all_careers(skill_vector):
    """
    Get prediction scores for ALL careers (useful for visualization).

    Returns:
        list of dicts sorted by confidence: [{"career": str, "confidence": float}, ...]
    """
    _load_artifacts()

    columns = _meta["skill_columns"]
    features = np.array([skill_vector.get(col, 0.0) for col in columns]).reshape(1, -1)

    probabilities = _model.predict_proba(features)[0]
    class_indices = np.argsort(probabilities)[::-1]

    results = []
    for idx in class_indices:
        career_name = _label_encoder.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        results.append({
            "career": career_name,
            "confidence": round(confidence, 4)
        })

    return results
