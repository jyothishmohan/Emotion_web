from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),          # Welcome page
    path('login/', views.login_view, name='login'),  # Login page
    path('register/', views.register_view, name='register'),
    path('predict/', views.index, name='predict'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('webcam_predict/', views.webcam_predict, name='webcam_predict'),
]