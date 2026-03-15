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
import base64
import anthropic

from collections import Counter
from tensorflow.keras.models import load_model
from django.core.files.base import ContentFile


# ======================================================
# LOAD MODELS (LOAD ONLY ONCE)
# ======================================================

face_model = None
text_predictor = None

face_emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


def get_face_model():
    global face_model

    if face_model is None:
        model_path = os.path.join(settings.BASE_DIR, 'emotion_model_improved.h5')

        if not os.path.exists(model_path):
            raise FileNotFoundError("emotion_model_improved.h5 not found")

        face_model = load_model(model_path)

    return face_model


def get_text_predictor():
    global text_predictor

    if text_predictor is None:
        text_predictor = TextEmotionPredictor()

    return text_predictor


# ======================================================
# CLAUDE-POWERED CHATBOT
# ======================================================

def generate_chatbot_response(emotion, user_message, conversation_history=None):
    """
    Uses the Anthropic Claude API to generate an intelligent, empathetic
    chatbot response based on the detected emotion and full conversation history.

    Args:
        emotion (str): The detected emotion label (e.g. "sadness", "joy")
        user_message (str): The user's latest message
        conversation_history (list): Prior turns as [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        str: Claude's reply
    """

    client = anthropic.Anthropic(api_key=getattr(settings, 'ANTHROPIC_API_KEY', None))

    system_prompt = f"""You are a warm, emotionally intelligent companion built into an Emotion Aware platform.
The user's current detected emotion is: **{emotion}**.

Your role:
- Acknowledge the user's emotion naturally — don't announce it robotically.
- Respond with genuine empathy, curiosity, and support.
- Keep replies concise (2–4 sentences) unless the user asks for more.
- Never give clinical diagnoses or tell the user to "just" do anything.
- If the emotion is positive (joy, love, surprise), match their energy and celebrate with them.
- If the emotion is negative (sadness, anger, fear, disgust), be gentle and validating.
- You may ask one thoughtful follow-up question to keep the conversation going.
- Do not mention that you are an AI unless directly asked."""

    messages_payload = []

    if conversation_history:
        messages_payload.extend(conversation_history)

    messages_payload.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=messages_payload,
        )
        return response.content[0].text

    except anthropic.APIError as e:
        # Graceful fallback so the app never crashes
        fallback_map = {
            "sadness": "I can hear that things feel difficult right now. I'm here with you.",
            "joy": "That's wonderful — sounds like things are going well!",
            "anger": "It sounds like something really frustrated you. Want to talk through it?",
            "fear": "Feeling scared is completely valid. You're not alone in this.",
            "love": "That's a beautiful feeling to be experiencing.",
            "surprise": "Wow, sounds like something unexpected happened!",
            "disgust": "That sounds really unpleasant. I'm sorry you had to experience that.",
        }
        return fallback_map.get(emotion.lower(), "Tell me more about how you're feeling.")


# ======================================================
# WELCOME PAGE
# ======================================================

def welcome(request):
    if request.user.is_authenticated:
        logout(request)
    return render(request, 'welcome.html')


# ======================================================
# REGISTER
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


# ======================================================
# LOGIN
# ======================================================

def login_view(request):
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


# ======================================================
# LOGOUT
# ======================================================

def logout_view(request):
    logout(request)
    return redirect('welcome')


# ======================================================
# FACE ONLY PREDICTION
# ======================================================

@login_required
def index(request):
    prediction = None
    confidence = None

    if request.method == 'POST':
        img_file = request.FILES.get('image')
        captured = request.POST.get('captured_image')

        if captured:
            format, imgstr = captured.split(';base64,')
            ext = format.split('/')[-1]
            img_file = ContentFile(base64.b64decode(imgstr), name='capture.' + ext)

        if img_file:
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
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face = gray[y:y + h, x:x + w]
            else:
                face = gray

            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            model_instance = get_face_model()
            preds = model_instance.predict(face, verbose=0)
            prediction = face_emotion_labels[np.argmax(preds)]
            confidence = float(np.max(preds)) * 100

            result.emotion = prediction
            result.face_confidence = confidence
            result.save()

        else:
            messages.error(request, "Upload or capture image")

    return render(request, 'index.html', {
        'prediction': prediction,
        'confidence': confidence,
    })


# ======================================================
# TEXT ONLY PREDICTION
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

            # Claude-powered first reply
            chatbot_reply = generate_chatbot_response(
                emotion=result['emotion'],
                user_message=text_input,
                conversation_history=None,
            )

            EmotionResult.objects.create(
                user=request.user,
                text_input=text_input,
                emotion=result['emotion'],
                prediction_type='text',
                text_confidence=confidence_percentage,
            )

            prediction_result = {
                'emotion': result['emotion'],
                'confidence': confidence_percentage,
                'all_probabilities': all_probs_percentage,
                'chatbot_reply': chatbot_reply,
            }

    return render(request, 'text_predict.html', {'prediction': prediction_result})


# ======================================================
# MULTIMODAL PREDICTION
# ======================================================

@login_required
def multimodal_predict(request):
    prediction_result = None

    if request.method == 'POST':
        img_file = request.FILES.get('image')
        captured = request.POST.get('captured_image')
        text_input = request.POST.get('text_input', '').strip()

        if captured:
            format, imgstr = captured.split(';base64,')
            ext = format.split('/')[-1]
            img_file = ContentFile(base64.b64decode(imgstr), name='capture.' + ext)

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
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face = gray[y:y + h, x:x + w]
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

            text_predictor_instance = get_text_predictor()
            text_result = text_predictor_instance.predict(text_input)
            text_emotion = text_result['emotion']
            text_confidence = float(text_result['confidence']) * 100

            combined = combine_emotion_predictions(
                normalize_emotion(face_emotion), face_confidence,
                normalize_emotion(text_emotion), text_confidence,
            )

            result.emotion = combined['final_emotion']
            result.face_confidence = face_confidence
            result.text_confidence = text_confidence
            result.combined_confidence = combined['combined_confidence']
            result.save()

            prediction_result = combined

        else:
            messages.error(request, "Provide image and text")

    return render(request, 'multimodal_predict.html', {'prediction': prediction_result})


# ======================================================
# DASHBOARD
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


# ======================================================
# TEXT API (JSON endpoint)
# ======================================================

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
                'probabilities': all_probs_percentage,
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


# ======================================================
# CHATBOT MESSAGE (AJAX — supports multi-turn history)
# ======================================================

@login_required
def chatbot_message(request):
    """
    Accepts POST JSON:
      {
        "message": "I feel really tired today",
        "emotion": "sadness",
        "history": [
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }
    Returns:
      {"emotion": "sadness", "reply": "..."}
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
            emotion = data.get("emotion", "neutral")
            history = data.get("history", [])  # full prior conversation turns

            if not user_message:
                return JsonResponse({"error": "Empty message"}, status=400)

            # Re-detect emotion from latest message for freshness
            predictor = get_text_predictor()
            result = predictor.predict(user_message)
            detected_emotion = result['emotion']

            # Use passed emotion if predictor result seems too generic
            final_emotion = detected_emotion if detected_emotion != "neutral" else emotion

            reply = generate_chatbot_response(
                emotion=final_emotion,
                user_message=user_message,
                conversation_history=history,
            )

            return JsonResponse({"emotion": final_emotion, "reply": reply})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

# ======================================================
# FREE CHATBOT PAGE + ENDPOINT
# ======================================================

@login_required
def chatbot_page(request):
    return render(request, 'chatbot.html')


@login_required
def chatbot_free(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data         = json.loads(request.body)
        user_message = data.get('message', '').strip()
        history      = data.get('history', [])

        if not user_message:
            return JsonResponse({'error': 'Empty message'}, status=400)

        predictor        = get_text_predictor()
        result           = predictor.predict(user_message)
        detected_emotion = result.get('emotion', 'neutral')

        EmotionResult.objects.create(
            user=request.user,
            text_input=user_message,
            emotion=detected_emotion,
            prediction_type='text',
            text_confidence=float(result.get('confidence', 0)) * 100,
        )

        client = anthropic.Anthropic(api_key=getattr(settings, 'ANTHROPIC_API_KEY', None))

        system_prompt = f"""You are a warm, emotionally intelligent companion on the Emotion Aware platform.
The user's latest detected emotion is: {detected_emotion}.

Your guidelines:
- Be genuinely empathetic, not clinical or scripted.
- Acknowledge the emotion naturally — never announce it robotically.
- Respond in 2–5 sentences unless the user needs more.
- If the emotion is positive, match their energy warmly.
- If negative, be validating and gentle — never dismissive.
- You may ask ONE thoughtful follow-up question to keep the dialogue going.
- If the user seems in genuine distress, gently suggest professional support without being preachy.
- Never break character or mention you are an AI unless the user directly asks."""

        messages_payload = list(history) + [{'role': 'user', 'content': user_message}]

        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=400,
            system=system_prompt,
            messages=messages_payload,
        )

        reply = response.content[0].text
        return JsonResponse({'emotion': detected_emotion, 'reply': reply})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ======================================================
# LIVE STREAM EMOTION (frame-by-frame AJAX)
# ======================================================

@login_required
def live_emotion(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data       = json.loads(request.body)
        frame_data = data.get('frame', '')

        if not frame_data:
            return JsonResponse({'error': 'No frame data'}, status=400)

        # Strip base64 header
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({'error': 'Could not decode frame'}, status=400)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return JsonResponse({'emotion': 'No face', 'confidence': 0.0, 'faces_found': 0})

        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]

        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        model_instance = get_face_model()
        preds          = model_instance.predict(face_roi, verbose=0)
        emotion        = face_emotion_labels[np.argmax(preds)]
        confidence     = float(np.max(preds)) * 100

        return JsonResponse({
            'emotion':     emotion,
            'confidence':  round(confidence, 2),
            'faces_found': len(faces),
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)