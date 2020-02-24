#imported to do the basic functions
from django.shortcuts import render, redirect

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

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

def detect(request):
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