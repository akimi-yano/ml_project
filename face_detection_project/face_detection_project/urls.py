"""face_detection_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('main_app.urls')),
    path('face_detection', include('face_detection_app.urls')),
    path('shape_detection', include('shape_detection_app.urls')),
    path('part_detection', include('part_detection_app.urls')),
    path('blink_detection', include('blink_detection_app.urls')),
    path('shape_detection', include('realtime.urls')),
    # path('admin/', admin.site.urls),
]