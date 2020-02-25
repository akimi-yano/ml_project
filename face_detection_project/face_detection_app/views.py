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

def detect_image(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES['image'])
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST['url']
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        # The  previous one was "flags=cv2.cv.CV_HAAR_SCALE_IMAGE" but I updated it to "flags=cv2.CASCADE_SCALE_IMAGE)"
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("IMAGE: " + str(grayscale))
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})
    # return a JSON response
    #return JsonResponse(data)
    for (startX, startY, endX, endY) in rects:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imwrite('temp.jpg', image)
    response = FileResponse(open('temp.jpg', 'rb'))
    os.remove('temp.jpg')
    return response

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

def detect_video(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method != "POST":
        # TODO handle correctly
        return None

    # check to see if an image was uploaded
    if request.FILES.get("video", None) is not None:
        # grab the uploaded video
        video = request.FILES['video']
        path = default_storage.save('tmp/somename.mp3', ContentFile(video.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        vs = FileVideoStream(tmp_file).start()
        fileStream = True
        time.sleep(1.0)
        # otherwise, assume that a URL was passed in
        # else:
        #     # grab the URL from the request
        #     url = request.POST['url']
        #     # if the URL is None, then return an error
        #     if url is None:
        #         data["error"] = "No URL provided."
        #         return JsonResponse(data)
        #     # load the image and convert
        #     image = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        # The  previous one was "flags=cv2.cv.CV_HAAR_SCALE_IMAGE" but I updated it to "flags=cv2.CASCADE_SCALE_IMAGE)"
        # loop over frames from the video stream
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
	# each loop iteration handles one frame in the video
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})
        # return a JSON response
        for (startX, startY, endX, endY) in rects:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 255), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 255), 1)
        
        # cv2.imwrite('temp.jpg', image)
        # response = FileResponse(open('temp.jpg', 'rb'))
        # os.remove('temp.jpg')
        # return response

        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("outpy.avi", fourcc, 30,
                (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")

        # break the image into its RGB components, then construct the
        # RGB representation of each frame individually
        (B, G, R) = cv2.split(frame)
        R = cv2.merge([zeros, zeros, R])
        G = cv2.merge([zeros, G, zeros])
        B = cv2.merge([B, zeros, zeros])

        # construct the final output frame, storing the original frame
        # at the top-left, the red channel in the top-right, the green
        # channel in the bottom-right, and the blue channel in the
        # bottom-left
        output = np.zeros((h, w, 3), dtype="uint8")
        output[0:h, 0:w] = frame
        # output[0:h, w:w * 2] = R
        # output[h:h * 2, w:w * 2] = G
        # output[h:h * 2, 0:w] = B

        # write the output frame to file
        writer.write(output)

    return_file = FileWrapper(open('outpy.avi', 'rb'))
    response = HttpResponse(return_file, content_type='video/avi')
    response['Content-Disposition'] = 'attachment; filename=my_video.avi'

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    writer.release()
    os.remove('outpy.avi')
    os.remove(tmp_file)
    return response
