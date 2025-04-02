from . import views
from django.urls import path
app_name= 'app'

urlpatterns = [
    path("", views.chat_view, name="chat_view"),
]
