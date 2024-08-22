from django import forms
from .models import UploadFile,Prompt
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import ConversationChat
# Define a new form inheriting from UserCreationForm
class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')
    first_name = forms.CharField(max_length=30)
    last_name = forms.CharField(max_length=30)
    qualification = forms.ChoiceField(choices=[('MD', 'Doctor of Medicine'), ('DO', 'Doctor of Osteopathic Medicine'), ('MBBS', 'Bachelor of Medicine, Bachelor of Surgery'), ('PhD', 'Doctor of Philosophy')], required=False)

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 'qualification')



class New_UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadFile
        fields = ['name', 'title', 'file','image','description']




class New_Prompt(forms.ModelForm):
    class Meta:
        model = Prompt  # Specify the model class
        fields = ['prompt_data']




class ConversationChatForm(forms.ModelForm):
    class Meta:
        model = ConversationChat
        fields = ['title']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'})
        }