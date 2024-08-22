from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinLengthValidator, MaxLengthValidator
from django.contrib.postgres.fields import ArrayField


from django.contrib.postgres.fields import JSONField  # If using PostgreSQL



class UploadFile(models.Model):
    name = models.CharField(max_length=100)
    title = models.CharField(max_length=100)
    description = models.TextField(max_length=600)  # New field for description
    image = models.ImageField(upload_to='images/')  # New field for image
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='uploads/')

    def __str__(self):
        return self.name

class Chunk(models.Model):
    document = models.ForeignKey(UploadFile, related_name='chunks', on_delete=models.CASCADE)
    content = models.TextField()
    embeddings = ArrayField(models.FloatField(), default=list)  # Store embeddings as an array of floats
    page_number = models.IntegerField()

    def __str__(self):
        return f"Chunk for {self.document.name} (Page {self.page_number})"





class Prompt(models.Model):
    document=models.ForeignKey(UploadFile,related_name='prompt',on_delete=models.CASCADE)
   
    prompt_data=models.TextField()



class ConversationChat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(UploadFile, on_delete=models.CASCADE)
    messages = models.JSONField()  # Store conversation messages as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    thread_id = models.IntegerField(default=0)
    title = models.CharField(max_length=255, blank=True, default='')
    

    def save(self, *args, **kwargs):
        if not self.title:
            self.title = self.generate_title()
        super().save(*args, **kwargs)

    def generate_title(self):
        if self.messages:
            first_user_message = next((msg['text'] for msg in self.messages if msg['source'] == 'user'), None)
            if first_user_message:
                return first_user_message[:15] + ('...' if len(first_user_message) > 15 else '')
        return 'Untitled Chat'

    def __str__(self):
        return f'ConversationChat(user={self.user}, document={self.document}, created_at={self.created_at}, chatid={self.id}, thread_id={self.thread_id})'




from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    username = models.CharField(max_length=150, blank=True, null=True, default='unknown')
    first_name = models.CharField(max_length=30, blank=True, null=True, default='unknown')
    last_name = models.CharField(max_length=150, blank=True, null=True, default='unknown')
    email = models.EmailField(blank=True, null=True, default='unknown')

    def save(self, *args, **kwargs):
        if self.user:
            self.username = self.user.username
            self.first_name = self.user.first_name
            self.last_name = self.user.last_name
            self.email = self.user.email
        super().save(*args, **kwargs)

    def __str__(self):
      return self.user.username

class UserSession(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    login_time = models.DateTimeField(null=True, blank=True)
    logout_time = models.DateTimeField(null=True, blank=True)
    time_spent = models.DurationField(null=True, blank=True)
    
    # New fields
    username = models.CharField(max_length=150, blank=True, null=True)
    first_name = models.CharField(max_length=30, blank=True, null=True)
    last_name = models.CharField(max_length=150, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if self.user_profile:
            self.username = self.user_profile.username
            self.first_name = self.user_profile.first_name
            self.last_name = self.user_profile.last_name
            self.email = self.user_profile.email
        if self.logout_time and self.login_time:
            self.time_spent = self.logout_time - self.login_time
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.username} - {self.login_time}"
