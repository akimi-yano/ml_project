from django.urls import path
from . import views

urlpatterns = [
    path('/image', views.part_image),
    path('/video', views.part_video)
]
