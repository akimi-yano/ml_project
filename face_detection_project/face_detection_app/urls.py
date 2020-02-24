from django.urls import path
from . import views

urlpatterns = [
    path('/image/detect', views.detect_image),
    path('/video/detect', views.detect_video)
]
