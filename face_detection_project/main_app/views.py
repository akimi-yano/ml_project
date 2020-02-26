from django.shortcuts import render
from .forms import UploadFileForm, UploadFileAndChoosePartForm, UploadVideoForm, UploadVideoAndChoosePartForm
def main(request):
    return render(request,"main_page.html")

def citations(request):
    return render(request,"citations.html")

def to_face_detection_image(request):
    form = UploadFileForm()
    context = {
        "form": form
    }
    return render(request, "to_face_detection.html", context)

def to_shape_detection_image(request):
    form = UploadFileForm()
    context = {
        "form": form
    }
    return render(request, "to_shape_detection.html", context)

def to_part_detection_image(request):
    form = UploadFileAndChoosePartForm()
    context = {
        "form": form
    }
    return render(request, "to_part_detection.html", context)

def to_blink_detection_video(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_blink_detection.html", context)

def to_shape_detection_realtime(request):
    return render(request,"to_shape_detection_realtime.html")


def to_face_detection_video(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_face_detection_video.html", context)

def to_shape_detection_video(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_shape_detection_video.html", context)

def to_part_detection_video(request):
    form = UploadVideoAndChoosePartForm()
    context = {
        "form": form
    }
    return render(request, "to_part_detection_video.html", context)

def to_wink_detection_video(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_wink_detection_video.html", context)

def to_sleepiness_detection_video(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_sleepiness_detection_video.html", context)

