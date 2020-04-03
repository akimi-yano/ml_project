#imported to do the basic functions
from django.shortcuts import render, redirect, HttpResponse

from pathlib import Path

#imported to do the test
import requests
import cv2
#imported to process image
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse
import numpy as np
import urllib
import json
import cv2
import os
import io


# imported additionally for video version
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from django.http import FileResponse
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from wsgiref.util import FileWrapper

# define the path to the face detector

cur_path = Path.cwd()
static_path = "face_detection_project/static/cascades/haarcascade_frontalface_default.xml"
FACE_DETECTOR_PATH = os.path.join(cur_path, static_path)

def detect_image(request):
    # check to see if this is a post request
    if request.method != "POST":
        return HttpResponse("Must upload picture")

    # check to see if an image was uploaded
    if request.FILES.get("image", None) is None:
        return HttpResponse("No image was uploaded")

    # grab the uploaded image
    image = _grab_image(stream=request.FILES['image'])
    processed_image = _process_face(image)

    error = None
    try:
        cv2.imwrite('temp.jpg', image)
        response = FileResponse(open('temp.jpg', 'rb'))
    except Exception as e:
        error = e
    finally:
        # make sure to delete files so that we don't run out of disk space
        os.remove('temp.jpg')

    if error:
        raise error
    return response


def detect_video(request):
    # check to see if this is a post request
    if request.method != "POST":
        return HttpResponse("Must upload video")

    # check to see if an image was uploaded
    if request.FILES.get("video", None) is None:
        return HttpResponse("No video was uploaded")

    # grab the uploaded video
    video = request.FILES['video']
    input_filename = None
    processed_filename = None
    try:
        path = default_storage.save("input", ContentFile(video.read()))
        input_filename = os.path.join(settings.MEDIA_ROOT, path)
        processed_filename = _process_face_video(input_filename)

        return_file = FileWrapper(open(processed_filename, 'rb'))
        response = HttpResponse(return_file, content_type='video/avi')
        response['Content-Disposition'] = 'attachment; filename=my_video.avi'
    except Exception as e:
        response = HttpResponse(f"Something went wrong when processing the video: {e}")
    finally:
        if input_filename:
            os.remove(input_filename)
        if processed_filename:
            os.remove(processed_filename)
    return response


def _process_face(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
    for (startX, startY, endX, endY) in rects:
        cv2.rectangle(image, (startX, startY), (endX, endY), (28, 12, 245), 2)
    return image


def _process_face_video(input_filename):
    vs = FileVideoStream(input_filename).start()
    fileStream = True
    time.sleep(1.0)
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
    # each loop iteration handles one frame in the video

    output_filename = None
    while True:
        if fileStream and not vs.more():
            break
        frame = vs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=450)

        if writer is None:
            (h, w) = frame.shape[:2]
            output_filename = "temp.avi"
            writer = cv2.VideoWriter(output_filename, fourcc, 30,
                (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")

        # detect face in this frame and save to processed_frame
        processed_frame = _process_face(frame)

        (B, G, R) = cv2.split(processed_frame)
        R = cv2.merge([zeros, zeros, R])
        G = cv2.merge([zeros, G, zeros])
        B = cv2.merge([B, zeros, zeros])

        output = np.zeros((h, w, 3), dtype="uint8")
        output[0:h, 0:w] = processed_frame

        writer.write(output)

    # do a bit of cleanup
    vs.stop()
    writer.release()
    return output_filename


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        # the older ver was using resp = urllib.urlopen(url) but I updated it to resp = urllib.request.urlopen(url)
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image