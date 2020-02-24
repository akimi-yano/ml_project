#imported to do general work
from django.shortcuts import render
from django.http import FileResponse
#imported to do shape predictor
from imutils import face_utils
import numpy as np
# import argparse
import imutils
import dlib
import cv2
import os

def shape(request):
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--shape-predictor", required=True,
	# 	help="path to facial landmark predictor")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(args["shape_predictor"])

	# get the full path of the predictor file; it is in this app, aka shape_detection_app/
	dir_path = os.path.dirname(os.path.realpath(__file__))
	predictor_path = os.path.join(dir_path, "shape_predictor_68_face_landmarks.dat")
	predictor = dlib.shape_predictor(predictor_path)

	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(args["image"])
	image = _grab_image(stream=request.FILES['image'])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)

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