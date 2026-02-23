from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import render
from .models import EmotionResult

import os
import json
import base64
import numpy as np
import cv2
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ======================================================
# 🔥 LOAD MODEL SAFELY (LOAD ONLY ONCE)
# ======================================================

model = None

def get_model():
    global model
    if model is None:
        model = load_model(os.path.join(settings.BASE_DIR, 'emotion_model_improved.h5'))
    return model


# 🔴 IMPORTANT: Replace with your real class order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


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
# 📷 IMAGE UPLOAD PREDICTION
# ======================================================

@login_required
def index(request):
    prediction = None

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']

        result = EmotionResult.objects.create(
            user=request.user,
            image=img_file,
            emotion="Processing..."
        )

        img_path = result.image.path

        # Preprocess image (48x48 grayscale)
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_instance = get_model()
        preds = model_instance.predict(img_array)
        prediction = emotion_labels[np.argmax(preds)]

        result.emotion = prediction
        result.save()

    return render(request, 'index.html', {'prediction': prediction})


# ======================================================
# 📊 DASHBOARD WITH CHART DATA
# ======================================================

@login_required
def dashboard(request):
    results = EmotionResult.objects.filter(user=request.user)

    emotions = [r.emotion for r in results]
    emotion_count = Counter(emotions)

    labels = list(emotion_count.keys())
    data = list(emotion_count.values())

    context = {
        'results': results,
        'labels': json.dumps(labels),
        'data': json.dumps(data),
    }

    return render(request, 'dashboard.html', context)


# ======================================================
# 🎥 WEBCAM PREDICTION
# ======================================================

from django.core.files.base import ContentFile
import uuid

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

            # 🔥 FACE DETECTION
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = gray[y:y+h, x:x+w]
            else:
                # If no face detected, use full image
                face = gray

            # Resize to model input
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Predict emotion
            model_instance = get_model()
            preds = model_instance.predict(face)
            prediction = emotion_labels[np.argmax(preds)]

            # 🔥 Save webcam image
            file_name = f"{uuid.uuid4()}.{ext}"
            image_file = ContentFile(decoded_img, name=file_name)

            EmotionResult.objects.create(
                user=request.user,
                image=image_file,
                emotion=prediction
            )

            return JsonResponse({'emotion': prediction})

        except Exception as e:
            print("Webcam Error:", e)
            return JsonResponse({'emotion': 'Error'})
def welcome(request):
    return render(request, 'welcome.html')        