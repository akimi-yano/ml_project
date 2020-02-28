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
    path('to_part_detection/video', views.to_part_detection_video),
    path('to_wink_detection/video', views.to_wink_detection_video),
    path('to_sleepiness_detection/video', views.to_sleepiness_detection_video),
    path('to_try_glasses/video', views.to_try_glasses_video),
    path('to_face_swap/video', views.to_face_swap_video),
    path('to_try_glasses/image', views.to_try_glasses_image),
    path('to_face_swap/image', views.to_face_swap_image)
 
    
]
