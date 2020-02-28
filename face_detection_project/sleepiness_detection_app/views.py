from django.shortcuts import render, redirect
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from django.http import FileResponse
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from imutils.video import FileVideoStream

def sleep_video(request):    

# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #     help="path to facial landmark predictor")
    # ap.add_argument("-a", "--alarm", type=str, default="",
    #     help="path alarm .WAV file")
    # ap.add_argument("-w", "--webcam", type=int, default=0,
    #     help="index of webcam on system")
    # args = vars(ap.parse_args())

    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.23
    EYE_AR_CONSEC_FRAMES = 48

    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    ALARM_ON = False

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    predictor_path = os.path.join(dir_path, "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)



    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    # print("[INFO] starting video stream thread...")
    # vs = VideoStream(src=args["webcam"]).start()
    # time.sleep(1.0)

    
    data = request.FILES['video']
    # data = request.POST['video']
    path = default_storage.save('tmp/somename.mp3', ContentFile(data.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, path)
    vs = FileVideoStream(tmp_file).start()
    
    
    fileStream = True
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    # fileStream = False
    time.sleep(1.0)
    
    
    
    
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
    
    # middle_mark = None
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        if fileStream and not vs.more():
            break

        frame = vs.read()
        if frame is None:
            break
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        
        
        # frame = vs.read()
        # frame = imutils.resize(frame, width=450)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        # rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True

                    # draw an alarm on the frame
                    cv2.putText(frame, "Sleepy alert!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y1, y2 = 100, 100 + sleepy_mark.shape[0]
                    x1, x2 = 10, 10 + sleepy_mark.shape[1]
                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (sleepy_mark_alpha_s * sleepy_mark[:, :, c] + sleepy_mark_alpha_l * frame[y1:y2, x1:x2, c])

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #     # show the frame
    #     cv2.imshow("Frame", frame)
    #     key = cv2.waitKey(1) & 0xFF
    
    #     # if the `q` key was pressed, break from the loop
    #     if key == ord("q"):
    #         break

    # # do a bit of cleanup
    # cv2.destroyAllWindows()
    # vs.stop()



        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("outpy.avi", fourcc, 30,
                (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")
            sleepy_mark = cv2.imread(request.POST['sleeping_image'], -1)
            sleepy_mark_alpha_s = sleepy_mark[:, :, 3] / 255.0
            sleepy_mark_alpha_l = 1.0 - sleepy_mark_alpha_s
            # right_mark = cv2.imread('wink_detection_app/star.png', -1)
            # right_mark_alpha_s = right_mark[:, :, 3] / 255.0
            # right_mark_alpha_l = 1.0 - right_mark_alpha_s
                
            # left_mark = cv2.imread('wink_detection_app/heart.png', -1)
            # left_mark_alpha_s = left_mark[:, :, 3] / 255.0
            # left_mark_alpha_l = 1.0 - left_mark_alpha_s
                
            # middle_mark = cv2.imread('wink_detection_app/kinoko.png', -1)
            # middle_mark_alpha_s = middle_mark[:, :, 3] / 255.0
            # middle_mark_alpha_l = 1.0 - middle_mark_alpha_s

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







def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear