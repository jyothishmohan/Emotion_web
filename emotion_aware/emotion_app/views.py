from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from .models import EmotionResult
from .utils import TextEmotionPredictor, combine_emotion_predictions, normalize_emotion

import os
import json
import base64
import numpy as np
import cv2
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.core.files.base import ContentFile
import uuid


# ======================================================
# 🔥 LOAD MODELS SAFELY (LOAD ONLY ONCE)
# ======================================================

face_model = None
text_predictor = None

def get_face_model():
    global face_model
    if face_model is None:
        face_model = load_model(os.path.join(settings.BASE_DIR, 'emotion_model_improved.h5'))
    return face_model

def get_text_predictor():
    global text_predictor
    if text_predictor is None:
        text_predictor = TextEmotionPredictor()
    return text_predictor


# Face emotion labels (from your CNN model)
face_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ======================================================
# 🔐 AUTHENTICATION (Your existing code)
# ======================================================

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect('register')

        User.objects.create_user(username=username, password=password)
        messages.success(request, "Account created successfully!")
        return redirect('login')

    return render(request, 'register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid credentials")

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


# ======================================================
# 📷 FACE-ONLY PREDICTION (Your existing code)
# ======================================================

@login_required
def index(request):
    prediction = None

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']

        result = EmotionResult.objects.create(
            user=request.user,
            image=img_file,
            emotion="Processing...",
            prediction_type='face'
        )

        img_path = result.image.path

        # Preprocess image (48x48 grayscale)
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_instance = get_face_model()
        preds = model_instance.predict(img_array)
        prediction = face_emotion_labels[np.argmax(preds)]
        confidence = float(np.max(preds))

        result.emotion = prediction
        result.face_confidence = confidence
        result.save()

    return render(request, 'index.html', {'prediction': prediction})


# ======================================================
# 📝 TEXT-ONLY PREDICTION (NEW)
# ======================================================

@login_required
def text_predict(request):
    """Predict emotion from text only"""
    prediction_result = None
    
    if request.method == 'POST':
        text_input = request.POST.get('text_input', '').strip()
        
        if text_input:
            # Get text predictor
            predictor = get_text_predictor()
            
            # Predict emotion
            result = predictor.predict(text_input)
            
            # Save to database
            db_result = EmotionResult.objects.create(
                user=request.user,
                text_input=text_input,
                emotion=result['emotion'],
                prediction_type='text',
                text_confidence=result['confidence']
            )
            
            prediction_result = {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities']
            }
    
    return render(request, 'text_predict.html', {'prediction': prediction_result})


# ======================================================
# 🎭 MULTIMODAL PREDICTION (NEW - Face + Text)
# ======================================================

@login_required
def multimodal_predict(request):
    """Predict emotion using both face and text"""
    prediction_result = None
    
    if request.method == 'POST':
        img_file = request.FILES.get('image')
        text_input = request.POST.get('text_input', '').strip()
        
        if img_file and text_input:
            # ===== FACE PREDICTION =====
            result = EmotionResult.objects.create(
                user=request.user,
                image=img_file,
                text_input=text_input,
                emotion="Processing...",
                prediction_type='multimodal'
            )
            
            img_path = result.image.path
            
            # Preprocess face image
            img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict face emotion
            face_model_instance = get_face_model()
            face_preds = face_model_instance.predict(img_array)
            face_emotion = face_emotion_labels[np.argmax(face_preds)]
            face_confidence = float(np.max(face_preds))
            
            # ===== TEXT PREDICTION =====
            text_predictor = get_text_predictor()
            text_result = text_predictor.predict(text_input)
            text_emotion = text_result['emotion']
            text_confidence = text_result['confidence']
            
            # ===== COMBINE PREDICTIONS =====
            # Normalize emotions for comparison
            face_emotion_normalized = normalize_emotion(face_emotion)
            text_emotion_normalized = normalize_emotion(text_emotion)
            
            combined = combine_emotion_predictions(
                face_emotion_normalized,
                face_confidence,
                text_emotion_normalized,
                text_confidence,
                face_weight=0.5,
                text_weight=0.5
            )
            
            # Update database
            result.emotion = combined['final_emotion']
            result.face_confidence = face_confidence
            result.text_confidence = text_confidence
            result.combined_confidence = combined['combined_confidence']
            result.save()
            
            prediction_result = {
                'final_emotion': combined['final_emotion'],
                'combined_confidence': combined['combined_confidence'],
                'face_emotion': face_emotion,
                'face_confidence': face_confidence,
                'text_emotion': text_emotion,
                'text_confidence': text_confidence,
                'agreement': combined['agreement'],
                'face_contribution': combined['face_contribution'],
                'text_contribution': combined['text_contribution'],
                'text_probabilities': text_result['all_probabilities']
            }
        else:
            messages.error(request, "Please provide both image and text")
    
    return render(request, 'multimodal_predict.html', {'prediction': prediction_result})


# ======================================================
# 📊 DASHBOARD (Updated)
# ======================================================

@login_required
def dashboard(request):
    results = EmotionResult.objects.filter(user=request.user)

    # Overall emotion distribution
    emotions = [r.emotion for r in results]
    emotion_count = Counter(emotions)

    # Prediction type distribution
    type_count = Counter([r.prediction_type for r in results])

    labels = list(emotion_count.keys())
    data = list(emotion_count.values())

    context = {
        'results': results,
        'labels': json.dumps(labels),
        'data': json.dumps(data),
        'type_labels': json.dumps(list(type_count.keys())),
        'type_data': json.dumps(list(type_count.values())),
        'total_predictions': results.count(),
        'face_predictions': results.filter(prediction_type='face').count(),
        'text_predictions': results.filter(prediction_type='text').count(),
        'multimodal_predictions': results.filter(prediction_type='multimodal').count(),
    }

    return render(request, 'dashboard.html', context)


# ======================================================
# 🎥 WEBCAM PREDICTION (Your existing code - can be enhanced)
# ======================================================

@login_required
def webcam_predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data['image']

            # Decode base64 image
            format, imgstr = image_data.split(';base64,')
            ext = format.split('/')[-1]
            decoded_img = base64.b64decode(imgstr)

            # Convert to OpenCV image
            nparr = np.frombuffer(decoded_img, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = gray[y:y+h, x:x+w]
            else:
                face = gray

            # Resize to model input
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict emotion
            model_instance = get_face_model()
            preds = model_instance.predict(face)
            prediction = face_emotion_labels[np.argmax(preds)]
            confidence = float(np.max(preds))

            # Save webcam image
            file_name = f"{uuid.uuid4()}.{ext}"
            image_file = ContentFile(decoded_img, name=file_name)

            EmotionResult.objects.create(
                user=request.user,
                image=image_file,
                emotion=prediction,
                prediction_type='face',
                face_confidence=confidence
            )

            return JsonResponse({
                'emotion': prediction,
                'confidence': confidence
            })

        except Exception as e:
            print("Webcam Error:", e)
            return JsonResponse({'emotion': 'Error', 'confidence': 0.0})

    return render(request, 'webcam.html')


# ======================================================
# 🏠 WELCOME PAGE
# ======================================================

def welcome(request):
    return render(request, 'welcome.html')


# ======================================================
# 🔥 API ENDPOINTS FOR AJAX (Optional)
# ======================================================

@login_required
def api_text_predict(request):
    """API endpoint for text prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            
            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)
            
            predictor = get_text_predictor()
            result = predictor.predict(text)
            
            return JsonResponse({
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'probabilities': result['all_probabilities']
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)