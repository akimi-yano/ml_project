from django.urls import path
from . import views

urlpatterns = [
    path('/video', views.try_glasses_video),
    path('/image', views.try_glasses_image)
]
