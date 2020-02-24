from django.urls import path
from . import views

urlpatterns = [
    path('/image', views.shape_image),
    path('/video', views.shape_video)
]
