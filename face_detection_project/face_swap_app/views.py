#imported to do the basic functions
from django.shortcuts import render, redirect, HttpResponse

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
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

def face_swap_image(request):
    # check to see if this is a post request
    if request.method != "POST":
        return HttpResponse("Must upload picture")

    # check to see if an image was uploaded
    if request.FILES.get("image", None) is None:
        return HttpResponse("No image was uploaded")

    # grab the uploaded image
    image = _grab_image(stream=request.FILES['image'])
    swap_mark = cv2.imread(request.POST['swap_image'], -1)
    processed_image = _process_face(image, swap_mark)

    try:
        cv2.imwrite('temp.jpg', image)
        response = FileResponse(open('temp.jpg', 'rb'))
    except Exception as e:
        response = HttpResponse(f"Something went wrong when processing the image: {e}")
    finally:
        # make sure to delete files so that we don't run out of disk space
        os.remove('temp.jpg')

    return response


def face_swap_video(request):
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
    error = None
    try:
        path = default_storage.save("input", ContentFile(video.read()))
        input_filename = os.path.join(settings.MEDIA_ROOT, path)
        processed_filename = _process_face_video(input_filename, request.POST['swap_image'])

        return_file = FileWrapper(open(processed_filename, 'rb'))
        response = HttpResponse(return_file, content_type='video/avi')
        response['Content-Disposition'] = 'attachment; filename=my_video.avi'
    except Exception as e:
        error = e
    finally:
        if input_filename:
            os.remove(input_filename)
        if processed_filename:
            os.remove(processed_filename)
            
    if error:
        raise error
    return response


def _process_face(image, swap_mark):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
    for (startX, startY, endX, endY) in rects:
        # cv2.rectangle(image, (startX, startY), (endX, endY), (28, 12, 245), 2)

        distance = endX - startX
        resized_swap_mark = imutils.resize(swap_mark, width=int(distance * 1.2))

        horizontal_shift = int(-distance * 0.08)
        vertical_shift = int(-distance * 0.2)
            # step 2: find starting point based on lStart
        y1, y2 = startY + vertical_shift, startY + vertical_shift + resized_swap_mark.shape[0]
        x1, x2 = startX + horizontal_shift, startX + horizontal_shift + resized_swap_mark.shape[1]

        # need to account for mark going out of frame
        y1_offset = max(y1, 0) - y1
        x1_offset = max(x1, 0) - x1
        y2_offset = y2 - min(y2, image.shape[0])
        x2_offset = x2 - min(x2, image.shape[1])

        resized_swap_mark_alpha_s = resized_swap_mark[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, 3] / 255.0
            # ((lEnd-lStart)/2-rStart)
        resized_swap_mark_alpha_l = 1.0 - resized_swap_mark_alpha_s
        for c in range(0, 3):
            image[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c] = (
                resized_swap_mark_alpha_s * resized_swap_mark[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, c] +
                resized_swap_mark_alpha_l * image[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c])
        # ((lEnd-lStart)/2-rStart)\

    return image


def _process_face_video(input_filename, swap_filename):
    vs = FileVideoStream(input_filename).start()
    fileStream = True
    time.sleep(1.0)
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
    # each loop iteration handles one frame in the video

    output_filename = None
    swap_mark = None
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
            
            swap_mark = cv2.imread(swap_filename, -1)

        # detect face in this frame and save to processed_frame
        processed_frame = _process_face(frame, swap_mark)

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