from django.shortcuts import render
from django.http import FileResponse
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os


def part(request):

    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #     help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    #     help="path to input image")
    # args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    predictor_path = os.path.join(dir_path, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)

    # load the input image, resize it, and convert it to grayscale
    image = _grab_image(stream=request.FILES['image'])
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name != request.POST['output_type']:
                continue
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", clone)
            # cv2.waitKey(0)

        # visualize all facial landmarks with a transparent overlay
        # output = face_utils.visualize_facial_landmarks(image, shape)
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)
        
        cv2.imwrite('temp.jpg', clone)
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