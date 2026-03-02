from django.db import models
from django.contrib.auth.models import User

class EmotionResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='emotions/', null=True, blank=True)
    text_input = models.TextField(null=True, blank=True)  # 👈 NEW
    emotion = models.CharField(max_length=50)
    prediction_type = models.CharField(max_length=20, default='face')  # 👈 NEW: 'face', 'text', 'multimodal'
    face_confidence = models.FloatField(null=True, blank=True)  # 👈 NEW
    text_confidence = models.FloatField(null=True, blank=True)  # 👈 NEW
    combined_confidence = models.FloatField(null=True, blank=True)  # 👈 NEW
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.emotion} ({self.prediction_type})"