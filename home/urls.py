from . import views
from django.urls import path
app_name= 'app'

urlpatterns = [
    path("", views.chat_view, name="chat_view"),
    path("clear_db_data", views.clear_db_data, name="clear_db_data"),
]
