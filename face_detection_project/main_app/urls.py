from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.main),
    path('to_face_detection/image', views.to_face_detection_image),
    path('to_shape_detection/image', views.to_shape_detection_image),
    path('to_part_detection/image', views.to_part_detection_image),
    path('to_blink_detection/video', views.to_blink_detection_video),
    path('citations', views.citations),
    path('to_face_detection/video', views.to_face_detection_video),
    path('to_shape_detection/video', views.to_shape_detection_video),
    path('to_part_detection/video', views.to_part_detection_video)
]
