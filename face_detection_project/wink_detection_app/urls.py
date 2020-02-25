from django.urls import path
from . import views

urlpatterns = [
    path('/video', views.wink_video)
]
