import os
import pickle
import re
import numpy as np
from django.conf import settings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ======================================================
# 📝 TEXT EMOTION PREDICTOR
# ======================================================

class TextEmotionPredictor:

    def __init__(self):
        base_path = os.path.join(settings.BASE_DIR, 'text_model')

        model_path = os.path.join(base_path, 'model.pkl')
        vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError("model.pkl not found inside text_model folder")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError("vectorizer.pkl not found inside text_model folder")

        self.model = pickle.load(open(model_path, 'rb'))
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # 🔥 6-Class Mapping
        self.label_map = {
            0: "Sad",
            1: "Happy",
            2: "Love",
            3: "Angry",
            4: "Fear",
            5: "Surprise"
        }

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text):
        words = text.split()
        words = [
            self.lemmatizer.lemmatize(w)
            for w in words
            if w not in self.stop_words and len(w) > 2
        ]
        return " ".join(words)

    def predict(self, text):
        cleaned = self.clean_text(text)
        processed = self.preprocess(cleaned)

        vector = self.vectorizer.transform([processed])
        probs = self.model.predict_proba(vector)[0]

        pred_class = np.argmax(probs)
        confidence = float(np.max(probs))

        return {
            "emotion": self.label_map[pred_class],
            "confidence": confidence,
            "all_probabilities": {
                self.label_map[i]: float(prob)
                for i, prob in enumerate(probs)
            }
        }


# ======================================================
# 🔄 NORMALIZE EMOTIONS FOR FUSION
# ======================================================

def normalize_emotion(emotion):
    """
    Align text emotions with face model labels
    """
    mapping = {
        "Love": "Happy",       # merge love into happy
        "Happy": "Happy",
        "Sad": "Sad",
        "Angry": "Angry",
        "Fear": "Fear",
        "Surprise": "Surprise",
        "Neutral": "Neutral",
        "Disgust": "Angry"     # optional mapping
    }

    return mapping.get(emotion, emotion)


# ======================================================
# 🎭 SMART FUSION FUNCTION
# ======================================================

def combine_emotion_predictions(
    face_emotion,
    face_confidence,
    text_emotion,
    text_confidence
):
    """
    Conflict-aware multimodal fusion logic
    """

    # Convert to float (safety)
    face_confidence = float(face_confidence)
    text_confidence = float(text_confidence)

    # --------------------------------------------------
    # CASE 1: Both modalities agree
    # --------------------------------------------------
    if face_emotion == text_emotion:

        # Give slightly higher weight to face (visual cue stronger)
        combined_confidence = (
            face_confidence * 0.6 +
            text_confidence * 0.4
        )

        return {
            "final_emotion": face_emotion,
            "face_emotion": face_emotion,
            "text_emotion": text_emotion,
            "face_confidence": face_confidence,
            "text_confidence": text_confidence,
            "combined_confidence": combined_confidence,
            "conflict": False
        }

    # --------------------------------------------------
    # CASE 2: Modalities disagree (Conflict detected)
    # --------------------------------------------------
    else:

        # Determine dominant modality
        if face_confidence >= text_confidence:
            dominant_emotion = face_emotion
        else:
            dominant_emotion = text_emotion

        # Reduce confidence because of disagreement
        average_confidence = (face_confidence + text_confidence) / 2

        # Penalize confidence due to conflict
        combined_confidence = average_confidence * 0.75

        return {
            "final_emotion": dominant_emotion,
            "face_emotion": face_emotion,
            "text_emotion": text_emotion,
            "face_confidence": face_confidence,
            "text_confidence": text_confidence,
            "combined_confidence": combined_confidence,
            "conflict": True
        }