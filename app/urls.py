
from django.urls import path
from . import views

urlpatterns = [
    # for account
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('',views.home,name='home'),
    #for profile
    
    path('redirect_to_dashboard/',views.redirect_to_dashboard,name='redirect_to_dashboard'),

    #for admin
    path('admin_dashboard/',views.admin_dashboard,name='admin_dashboard'),
    path('all_docs/',views.all_docs,name='list_doc'),
    path('edit_doc/<int:doc_id>',views.edit_doc,name='edit_doc'),
    path('edit-prompt/<int:prompt_id>/', views.edit_prompt, name='edit_prompt'),
    path('delete-prompt/<int:prompt_id>/', views.delete_prompt, name='delete_prompt'),

    path('confirmDelete/<int:doc_id>/', views.confirm_delete, name='delete_doc'),
    path('admin_new_upload/',views.admin_new_upload,name='admin_new_upload'),
    # for user
    path('user_dashboard/',views.user_dashboard,name='user_dashboard'),
    path('common_chat-interface/', views.common_chat_interface, name='common_chat_interface'),


    
    
    path('chat-interface/<int:document_id>/', views.chat_interface, name='chat_interface'),
    
  
    path('save-chat/', views.save_chat, name='save_chat'),
    # path('start_new_chat/', views.start_new_chat, name='start_new_chat'),
    path('clear_chat/', views.clear_chat, name='clear_chat'),

    path('copy_to_clipboard/',views.copy_to_clipboard,name='copy_to_clipboard'),
    path('document/<int:document_id>/', views.document_viewer, name='document_viewer'),

    path('export-export_conversation_doc/', views.export_conversation_doc, name='export_conversation_doc'),
    path('export_conversation_pdf/', views.export_conversation_pdf, name='export_conversation_pdf'),
    path('export_conversation_ppt/',views.export_conversation_ppt, name='export_conversation_ppt'),
    path('export_conversation_json/',views.export_conversation_json, name='export_conversation_json'),
    path('export_conversation_audio/',views.export_conversation_audio, name='export_conversation_audio'),
    # for common chat
    path('common_export-common_export_conversation_doc/', views.common_export_conversation_doc, name='common_export_conversation_doc'),
    path('common_export_conversation_pdf/', views.common_export_conversation_pdf, name='common_export_conversation_pdf'),
    path('common_export_conversation_ppt/',views.common_export_conversation_ppt, name='common_export_conversation_ppt'),
    path('common_export_conversation_json/',views.common_export_conversation_json, name='common_export_conversation_json'),

    path('common_export_conversation_audio/',views.common_export_conversation_audio, name='common_export_conversation_audio'),

]
