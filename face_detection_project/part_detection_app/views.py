from django.shortcuts import render, HttpResponse
from django.http import FileResponse
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from wsgiref.util import FileWrapper
from imutils.video import FileVideoStream
import time

from pathlib import Path


def part_image(request):

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
    
    cur_path = str(Path.cwd())
    shape_predictor_path = "face_detection_project/static/shape_predictor_68_face_landmarks.dat"
    predictor_path = os.path.join(cur_path, shape_predictor_path)
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

def part_video(request):
    
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
    
    cur_path = str(Path.cwd())
    shape_predictor_path = "face_detection_project/static/shape_predictor_68_face_landmarks.dat"
    predictor_path = os.path.join(cur_path, shape_predictor_path)
    predictor = dlib.shape_predictor(predictor_path)
    
    
    if request.FILES.get("video", None) is not None:
        # grab the uploaded video
        video = request.FILES['video']
        path = default_storage.save('tmp/somename.mp3', ContentFile(video.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        vs = FileVideoStream(tmp_file).start()
        fileStream = True
        time.sleep(1.0)
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
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
                clone = frame.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi = frame[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                
                        
                # y1, y2 = 100, 100 + roi[0]
                # x1, x2 = 350, 350 + roi[1]
                # clone[y1:y2, x1:x2, c] = (roi_alpha_s * roi[:, :, c] +
                #                                     roi_alpha_l * frame[y1:y2, x1:x2, c])

                # show the particular face part
                # cv2.imshow("ROI", roi)
                # cv2.imshow("Image", clone)
                # cv2.waitKey(0)

            # visualize all facial landmarks with a transparent overlay
            # output = face_utils.visualize_facial_landmarks(image, shape)
            # cv2.imshow("Image", output)
            # cv2.waitKey(0)
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            (video_height, video_width) = clone.shape[:2]
            writer = cv2.VideoWriter("outpy.avi", fourcc, 30,
                (video_width, video_height), True)
            zeros = np.zeros((video_height, video_width), dtype="uint8")
            
            # roi_alpha_s = roi[:, :, 3] / 255.0
            # roi_alpha_l = 1.0 - roi_alpha_s

        # break the image into its RGB components, then construct the
        # RGB representation of each frame individually
        (B, G, R) = cv2.split(clone)
        R = cv2.merge([zeros, zeros, R])
        G = cv2.merge([zeros, G, zeros])
        B = cv2.merge([B, zeros, zeros])

        output = np.zeros((video_height, video_width, 3), dtype="uint8")
        output[0:video_height, 0:video_width] = clone
        
        

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


