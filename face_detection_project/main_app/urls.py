from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main),
    path('to_face_detection/image', views.to_face_detection),
    path('to_shape_detection/image', views.to_shape_detection),
    path('to_part_detection/image', views.to_part_detection),
    path('to_blink_detection/video', views.to_blink_detection),
    path('to_shape_detection/realtime', views.to_shape_detection_realtime),
    path('citations', views.citations)
]
