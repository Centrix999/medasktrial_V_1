

from django.contrib import admin
from django.contrib.sessions.models import Session
import json
from django.utils.safestring import mark_safe
from app.models import UploadFile, Chunk,ConversationChat,UserProfile,UserSession

# Register your models here.
admin.site.register(UploadFile)
admin.site.register(Chunk)

class SessionAdmin(admin.ModelAdmin):
    list_display = ['session_key', 'session_data', 'expire_date']
    readonly_fields = ['session_key', 'session_data', 'expire_date']
    
    def session_data(self, obj):
        # Decode the session data
        data = obj.get_decoded()
        # Pretty-print the session data as JSON
        return mark_safe('<pre>{}</pre>'.format(json.dumps(data, indent=4)))

# Register the Session model with the custom admin view
admin.site.register(Session, SessionAdmin)
admin.site.register(UserProfile)
admin.site.register(ConversationChat)
admin.site.register(UserSession)