from django.urls import path

from . import views

urlpatterns = [
    path("", views.chat_completion, name="chat_completion"),
]
