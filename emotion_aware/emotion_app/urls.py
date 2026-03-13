from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),

    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    path('dashboard/', views.dashboard, name='dashboard'),

    path('face/', views.index, name='index'),
    path('text/', views.text_predict, name='text_predict'),
    path('multimodal/', views.multimodal_predict, name='multimodal_predict'),

    path('api/text/', views.api_text_predict, name='api_text_predict'),
    path('chatbot_message/', views.chatbot_message, name='chatbot_message'),
    path('chatbot/', views.chatbot_page, name='chatbot'),
    path('chatbot/message/', views.chatbot_free, name='chatbot_free'),
    path('live-emotion/', views.live_emotion, name='live_emotion'),
]