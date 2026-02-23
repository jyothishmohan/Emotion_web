from django.contrib import admin
from .models import EmotionResult

class EmotionResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'emotion', 'created_at')
    list_filter = ('emotion', 'created_at')
    search_fields = ('user__username', 'emotion')
    
    

admin.site.register(EmotionResult, EmotionResultAdmin)
