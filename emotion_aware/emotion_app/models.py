from django.db import models
from django.contrib.auth.models import User


class EmotionResult(models.Model):
    PREDICTION_TYPE_CHOICES = [
        ('face', 'Face Only'),
        ('text', 'Text Only'),
        ('multimodal', 'Face + Text'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Input data
    image = models.ImageField(upload_to='emotion_images/', null=True, blank=True)
    text_input = models.TextField(null=True, blank=True)
    
    # Results
    emotion = models.CharField(max_length=50)
    prediction_type = models.CharField(
        max_length=20, 
        choices=PREDICTION_TYPE_CHOICES, 
        default='face'
    )
    
    # Confidence scores
    face_confidence = models.FloatField(default=0.0)
    text_confidence = models.FloatField(default=0.0)
    combined_confidence = models.FloatField(default=0.0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.emotion} ({self.prediction_type})"
    
    def get_confidence_display(self):
        """Return appropriate confidence based on prediction type"""
        if self.prediction_type == 'face':
            return f"{self.face_confidence * 100:.1f}%"
        elif self.prediction_type == 'text':
            return f"{self.text_confidence * 100:.1f}%"
        else:
            return f"{self.combined_confidence * 100:.1f}%"
    
    class Meta:
        ordering = ['-created_at']