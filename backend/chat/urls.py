from django.urls import path

from . import views

urlpatterns = [
    path("sessions/<uuid:pk>/", views.session_detail, name="session_detail"),
    path("sessions/", views.list_sessions, name="list_sessions"),
    path("", views.chat_completion, name="chat_completion"),
]
