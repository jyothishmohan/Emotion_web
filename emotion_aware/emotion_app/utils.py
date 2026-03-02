import os
import pickle
import re
import numpy as np
from django.conf import settings
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.corpus import stopwords

# ======================================================
# 🔥 LOAD TEXT EMOTION MODEL (SINGLETON PATTERN)
# ======================================================

class TextEmotionPredictor:
    _instance = None
    _model = None
    _vectorizer = None
    _label_encoder = None
    _lemmatizer = None
    _stop_words = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextEmotionPredictor, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        """Load all required models and preprocessors"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'text_emotion_model')
            
            # Load model
            with open(os.path.join(model_dir, 'text_emotion_model.pkl'), 'rb') as f:
                self._model = pickle.load(f)
            
            # Load vectorizer
            with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
                self._vectorizer = pickle.load(f)
            
            # Load label encoder
            with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
                self._label_encoder = pickle.load(f)
            
            # Load preprocessing config
            with open(os.path.join(model_dir, 'preprocessing_config.pkl'), 'rb') as f:
                config = pickle.load(f)
                self._stop_words = set(config['stop_words'])
            
            # Initialize lemmatizer
            self._lemmatizer = WordNetLemmatizer()
            
            print("✓ Text emotion model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading text emotion model: {e}")
            raise

    def clean_text(self, text):
        """Clean and preprocess text data"""
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_text(self, text):
        """Remove stopwords and apply lemmatization"""
        words = text.split()
        words = [self._lemmatizer.lemmatize(word) for word in words 
                 if word not in self._stop_words and len(word) > 2]
        return ' '.join(words)

    def predict(self, text):
        """
        Predict emotion from text
        
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        try:
            # Preprocess
            cleaned = self.clean_text(text)
            processed = self.preprocess_text(cleaned)
            
            if not processed.strip():
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'all_probabilities': {}
                }
            
            # Vectorize
            vectorized = self._vectorizer.transform([processed])
            
            # Predict
            prediction = self._model.predict(vectorized)[0]
            emotion = self._label_encoder.inverse_transform([prediction])[0]
            
            # Get probabilities
            probabilities = self._model.predict_proba(vectorized)[0]
            confidence = float(max(probabilities))
            
            # Create probability dictionary
            prob_dict = {
                self._label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_probabilities': prob_dict
            }
            
        except Exception as e:
            print(f"Error in text prediction: {e}")
            return {
                'emotion': 'error',
                'confidence': 0.0,
                'all_probabilities': {}
            }

    def get_available_emotions(self):
        """Get list of all emotions the model can predict"""
        return list(self._label_encoder.classes_)


# ======================================================
# 🎯 MULTIMODAL EMOTION FUSION
# ======================================================

def combine_emotion_predictions(face_emotion, face_confidence, text_emotion, text_confidence, 
                                face_weight=0.5, text_weight=0.5):
    """
    Combine face and text emotion predictions
    
    Args:
        face_emotion: predicted emotion from face
        face_confidence: confidence score for face prediction (0-1)
        text_emotion: predicted emotion from text
        text_confidence: confidence score for text prediction (0-1)
        face_weight: weight for face prediction (default 0.5)
        text_weight: weight for text prediction (default 0.5)
    
    Returns:
        dict: {
            'final_emotion': str,
            'combined_confidence': float,
            'face_contribution': float,
            'text_contribution': float,
            'agreement': bool
        }
    """
    
    # Normalize weights
    total_weight = face_weight + text_weight
    face_weight = face_weight / total_weight
    text_weight = text_weight / total_weight
    
    # Check if emotions match
    agreement = (face_emotion.lower() == text_emotion.lower())
    
    if agreement:
        # If both agree, combine confidences
        combined_confidence = (face_confidence * face_weight + text_confidence * text_weight)
        final_emotion = face_emotion
    else:
        # If they disagree, choose the one with higher weighted confidence
        face_score = face_confidence * face_weight
        text_score = text_confidence * text_weight
        
        if face_score > text_score:
            final_emotion = face_emotion
            combined_confidence = face_score
        else:
            final_emotion = text_emotion
            combined_confidence = text_score
    
    return {
        'final_emotion': final_emotion,
        'combined_confidence': float(combined_confidence),
        'face_contribution': float(face_confidence * face_weight),
        'text_contribution': float(text_confidence * text_weight),
        'agreement': agreement,
        'face_emotion': face_emotion,
        'text_emotion': text_emotion,
        'face_confidence': float(face_confidence),
        'text_confidence': float(text_confidence)
    }


# ======================================================
# 🎨 EMOTION MAPPING (Align face and text emotions)
# ======================================================

# Map different emotion names to standard ones
EMOTION_MAPPING = {
    # Face emotions (your CNN model)
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'joy',
    'sad': 'sadness',
    'surprise': 'surprise',
    'neutral': 'neutral',
    
    # Text emotions (from dataset)
    'anger': 'anger',
    'joy': 'joy',
    'sadness': 'sadness',
    'fear': 'fear',
    'love': 'joy',  # Map love to joy
    'surprise': 'surprise',
}

def normalize_emotion(emotion):
    """Normalize emotion names for comparison"""
    emotion_lower = emotion.lower()
    return EMOTION_MAPPING.get(emotion_lower, emotion_lower)