from django.urls import path
from . import views

urlpatterns = [
    path('/video', views.blink_video)
]
