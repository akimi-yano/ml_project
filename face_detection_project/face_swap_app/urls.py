from django.urls import path
from . import views

urlpatterns = [
    path('/video', views.face_swap_video)
]
