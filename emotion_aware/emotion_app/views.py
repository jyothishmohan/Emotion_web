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
import numpy as np
import cv2
from collections import Counter
from tensorflow.keras.models import load_model


# ======================================================
# 🔥 LOAD MODELS SAFELY (LOAD ONLY ONCE)
# ======================================================

face_model = None
text_predictor = None

face_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def get_face_model():
    global face_model
    if face_model is None:
        model_path = os.path.join(settings.BASE_DIR, 'emotion_model_improved.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError("emotion_model_improved.h5 not found in BASE_DIR")
        face_model = load_model(model_path)
    return face_model


def get_text_predictor():
    global text_predictor
    if text_predictor is None:
        text_predictor = TextEmotionPredictor()
    return text_predictor


# ======================================================
# 🌟 WELCOME PAGE
# ======================================================

def welcome(request):
    # Always logout when visiting welcome page
    if request.user.is_authenticated:
        logout(request)
    return render(request, 'welcome.html')


# ======================================================
# 🔐 AUTHENTICATION
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
    # Force logout if already logged in
    if request.user.is_authenticated:
        logout(request)

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
    return redirect('welcome')


# ======================================================
# 📷 FACE ONLY PREDICTION
# ======================================================

@login_required
def index(request):
    prediction = None
    confidence = None

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']

        result = EmotionResult.objects.create(
            user=request.user,
            image=img_file,
            emotion="Processing...",
            prediction_type='face'
        )

        img_path = result.image.path

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
        else:
            face = gray

        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        model_instance = get_face_model()
        preds = model_instance.predict(face, verbose=0)

        prediction = face_emotion_labels[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100  # Convert to %

        result.emotion = prediction
        result.face_confidence = confidence
        result.save()

    return render(request, 'index.html', {
        'prediction': prediction,
        'confidence': confidence
    })


# ======================================================
# 📝 TEXT ONLY PREDICTION
# ======================================================

@login_required
def text_predict(request):
    prediction_result = None

    if request.method == 'POST':
        text_input = request.POST.get('text_input', '').strip()

        if text_input:
            predictor = get_text_predictor()
            result = predictor.predict(text_input)

            confidence_percentage = float(result['confidence']) * 100

            all_probs_percentage = {}
            if 'all_probabilities' in result:
                for emotion, prob in result['all_probabilities'].items():
                    all_probs_percentage[emotion] = float(prob) * 100

            EmotionResult.objects.create(
                user=request.user,
                text_input=text_input,
                emotion=result['emotion'],
                prediction_type='text',
                text_confidence=confidence_percentage
            )

            prediction_result = {
                'emotion': result['emotion'],
                'confidence': confidence_percentage,
                'all_probabilities': all_probs_percentage
            }

    return render(request, 'text_predict.html', {
        'prediction': prediction_result
    })


# ======================================================
# 🎭 MULTIMODAL PREDICTION
# ======================================================

@login_required
def multimodal_predict(request):
    prediction_result = None

    if request.method == 'POST':
        img_file = request.FILES.get('image')
        text_input = request.POST.get('text_input', '').strip()

        if img_file and text_input:

            result = EmotionResult.objects.create(
                user=request.user,
                image=img_file,
                text_input=text_input,
                emotion="Processing...",
                prediction_type='multimodal'
            )

            img_path = result.image.path

            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face = gray[y:y+h, x:x+w]
            else:
                face = gray

            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            face_model_instance = get_face_model()
            face_preds = face_model_instance.predict(face, verbose=0)

            face_emotion = face_emotion_labels[np.argmax(face_preds)]
            face_confidence = float(np.max(face_preds)) * 100

            text_predictor = get_text_predictor()
            text_result = text_predictor.predict(text_input)

            text_emotion = text_result['emotion']
            text_confidence = float(text_result['confidence']) * 100

            combined = combine_emotion_predictions(
                normalize_emotion(face_emotion),
                face_confidence,
                normalize_emotion(text_emotion),
                text_confidence
            )

            result.emotion = combined['final_emotion']
            result.face_confidence = face_confidence
            result.text_confidence = text_confidence
            result.combined_confidence = combined['combined_confidence']
            result.save()

            prediction_result = combined

        else:
            messages.error(request, "Please provide both image and text")

    return render(request, 'multimodal_predict.html', {
        'prediction': prediction_result
    })


# ======================================================
# 📊 DASHBOARD
# ======================================================

@login_required
def dashboard(request):
    results = EmotionResult.objects.filter(user=request.user)

    emotions = [r.emotion for r in results if r.emotion != "Processing..."]
    emotion_count = Counter(emotions)
    type_count = Counter([r.prediction_type for r in results])

    context = {
        'results': results,
        'labels': json.dumps(list(emotion_count.keys())),
        'data': json.dumps(list(emotion_count.values())),
        'type_labels': json.dumps(list(type_count.keys())),
        'type_data': json.dumps(list(type_count.values())),
        'total_predictions': results.count(),
        'face_predictions': results.filter(prediction_type='face').count(),
        'text_predictions': results.filter(prediction_type='text').count(),
        'multimodal_predictions': results.filter(prediction_type='multimodal').count(),
    }

    return render(request, 'dashboard.html', context)

@login_required
def api_text_predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()

            if not text:
                return JsonResponse({'error': 'No text provided'}, status=400)

            predictor = get_text_predictor()
            result = predictor.predict(text)

            confidence_percentage = float(result['confidence']) * 100

            all_probs_percentage = {}
            if 'all_probabilities' in result:
                for emotion, prob in result['all_probabilities'].items():
                    all_probs_percentage[emotion] = float(prob) * 100

            return JsonResponse({
                'emotion': result['emotion'],
                'confidence': confidence_percentage,
                'probabilities': all_probs_percentage
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)