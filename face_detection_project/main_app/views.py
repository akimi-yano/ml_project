from django.shortcuts import render
from .forms import UploadFileForm, UploadFileAndChoosePartForm, UploadVideoForm

def main(request):
    return render(request,"main_page.html")

def citations(request):
    return render(request,"citations.html")

def to_face_detection(request):
    form = UploadFileForm()
    context = {
        "form": form
    }
    return render(request, "to_face_detection.html", context)

def to_shape_detection(request):
    form = UploadFileForm()
    context = {
        "form": form
    }
    return render(request, "to_shape_detection.html", context)

def to_part_detection(request):
    form = UploadFileAndChoosePartForm()
    context = {
        "form": form
    }
    return render(request, "to_part_detection.html", context)

def to_blink_detection(request):
    form = UploadVideoForm()
    context = {
        "form": form
    }
    return render(request, "to_blink_detection.html", context)

def to_shape_detection_realtime(request):
    return render(request,"to_shape_detection_realtime.html")


