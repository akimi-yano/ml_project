from django.shortcuts import render, HttpResponse

from pathlib import Path

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from django.http import FileResponse
from imutils import face_utils
import numpy as np
# import argparse
import imutils
import time
import dlib
import cv2
import os
#experiement: to put emojis
# from PIL import Image, ImageFont, ImageDraw

from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from wsgiref.util import FileWrapper

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

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
def wink_video(request):
    EYE_AR_THRESH = 0.21
    # EYE_AR_THRESH2 = 0.23
    EYE_AR_CONSEC_FRAMES = 1
    

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    LEFT_COUNTER = 0
    RIGHT_COUNTER = 0
    TOTAL = 0
    BLINK_SHOW = 0
    LEFTWINK_SHOW = 0
    RIGHTWINK_SHOW = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])

    cur_path = str(Path.cwd())
    shape_predictor_path = "face_detection_project/static/shape_predictor_68_face_landmarks.dat"
    predictor_path = os.path.join(cur_path, shape_predictor_path)
    predictor = dlib.shape_predictor(predictor_path)

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    # print("[INFO] starting video stream thread...")
    # vs = _grab_video(stream=request.FILES['video'])
    data = request.FILES['video']
    path = default_storage.save('tmp/somename.mp3', ContentFile(data.read()))
    tmp_file = os.path.join(settings.MEDIA_ROOT, path)
    vs = FileVideoStream(tmp_file).start()
    # else:
    #     return HttpResponse("No Video was uploaded")
    # vs = FileVideoStream(args["video"]).start()
    fileStream = True
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    # fileStream = False
    time.sleep(1.0)

    # loop over frames from the video stream
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    writer = None
    right_mark = None
    left_mark = None
    middle_mark = None
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

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

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
            #this is to show the outlines of eyes
            # cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 255), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 255), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            # if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            elif leftEAR >= EYE_AR_THRESH or rightEAR >= EYE_AR_THRESH:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if LEFTWINK_SHOW == 0 and RIGHTWINK_SHOW == 0 and COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    BLINK_SHOW = 5
                # set star countdown variable to 10 frames

                # reset the eye frame counter
                COUNTER = 0
            
            if leftEAR < EYE_AR_THRESH and rightEAR >= EYE_AR_THRESH:
                LEFT_COUNTER += 1
            elif leftEAR >= EYE_AR_THRESH or rightEAR < EYE_AR_THRESH:
                if LEFT_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    LEFTWINK_SHOW = 5
                LEFT_COUNTER = 0
                # if leftEAR < EYE_AR_THRESH2 and rightEAR < EYE_AR_THRESH2:
                #     COUNTER += 1
                
            if rightEAR < EYE_AR_THRESH and leftEAR >= EYE_AR_THRESH:
                RIGHT_COUNTER += 1
            elif rightEAR >= EYE_AR_THRESH or leftEAR < EYE_AR_THRESH:
                if RIGHT_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    RIGHTWINK_SHOW = 5
                RIGHT_COUNTER = 0
                # if leftEAR < EYE_AR_THRESH2 and rightEAR < EYE_AR_THRESH2:
                #     COUNTER += 1
            

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # cv2.putText(frame, "RightEAR: {:.2f}".format(rightEAR), (10, 200),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # cv2.putText(frame, "LeftEAR: {:.2f}".format(leftEAR), (300, 200),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if BLINK_SHOW > 0:
                # cv2.putText(frame, "BLINKING!", (150, 30),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y1, y2 = 20, 20 + middle_mark.shape[0]
                x1, x2 = 190, 190 + middle_mark.shape[1]
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (middle_mark_alpha_s * middle_mark[:, :, c] +
                                              middle_mark_alpha_l * frame[y1:y2, x1:x2, c])
                BLINK_SHOW -= 1
            if LEFTWINK_SHOW > 0:
                # cv2.putText(frame, "LEFT WINK!", (300, 100),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.circle(frame, (400,75), 30, (0,255,255), -1)
                y1, y2 = 100, 100 + left_mark.shape[0]
                x1, x2 = 350, 350 + left_mark.shape[1]
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (left_mark_alpha_s * left_mark[:, :, c] +
                                              left_mark_alpha_l * frame[y1:y2, x1:x2, c])
                    
                LEFTWINK_SHOW -= 1
            if RIGHTWINK_SHOW > 0:
                # cv2.putText(frame, "RIGHT WINK!", (10, 100),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # p1 = (45, 50) 
                # p2 = (105, 50) 
                # p3 = (75, 100)
                # cv2.circle(frame, p1, 2, (0,0,255), -1)
                # cv2.circle(frame, p2, 2, (0,0,255), -1)
                # cv2.circle(frame, p3, 2, (0,0,255), -1)
                # triangle_cnt = np.array([p1, p2, p3])
                # cv2.drawContours(frame, [triangle_cnt], 0, (0,0,255), -1)
                
                y1, y2 = 100, 100 + right_mark.shape[0]
                x1, x2 = 80, 80 + right_mark.shape[1]
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (right_mark_alpha_s * right_mark[:, :, c] +
                                              right_mark_alpha_l * frame[y1:y2, x1:x2, c])
                RIGHTWINK_SHOW -= 1
                
            #test
            
            # @@@this will show if its left wink
            # cv2.circle(frame, (400,75), 30, (0,255,255), -1)
            
            
            
            # cv.circle(img,(447,63), 63, (0,0,255), -1)
            # Let's define four points
            # pts = np.array( [[150, 100], [100, 200], [200, 200]], np.int32)
            # # Let's now reshape our points in form  required by polylines
            # pts = pts.reshape((-1,1,2))
            # cv2.polylines(frame, [pts],  True, (0,0,255), -1)
            
            # @@@this will show if its right wink
            # p1 = (45, 50) 
            # p2 = (105, 50) 
            # p3 = (75, 100) 
            # cv2.circle(frame, p1, 2, (0,0,255), -1)
            # cv2.circle(frame, p2, 2, (0,0,255), -1)
            # cv2.circle(frame, p3, 2, (0,0,255), -1)
            # # We can put the three points into an array and draw as a contour:
            # triangle_cnt = np.array( [p1, p2, p3] )
            # cv2.drawContours(frame, [triangle_cnt], 0, (0,0,255), -1)
            
            # # Drawing the triangle with the help of lines 
            # #  on the black window With given points  
            # # cv2.line is the inbuilt function in opencv library 
            # cv2.line(frame, p1, p2, (255, 0, 0), 3) 
            # cv2.line(frame, p2, p3, (255, 0, 0), 3) 
            # cv2.line(frame, p1, p3, (255, 0, 0), 3) 
            
            
            # pts = np.array([[50, 50],[100, 50],[75, 100]], np.int32)
            # pts = pts.reshape((-1,1,2))
            # cv2.polylines(frame,[pts],True,(0,255,255))
            
            # image = np.ones((300, 300, 3), np.uint8) * 255

            
            
            
            # finding centroid using the following formula 
            # (X, Y) = (x1 + x2 + x3//3, y1 + y2 + y3//3)  
            # centroid = ((p1[0]+p2[0]+p3[0])//3, (p1[1]+p2[1]+p3[1])//3) 
            
            # # Drawing the centroid on the window   
            # cv2.circle(frame, centroid, 4, (0, 255, 0)) 

            # pt1 = (150, 100)
            # pt2 = (100, 200)
            # pt3 = (200, 200)

            # cv2.circle(frame, pt1, 2, (0,0,255), -1)
            # cv2.circle(frame, pt2, 2, (0,0,255), -1)
            # cv2.circle(frame, pt3, 2, (0,0,255), -1)

            # cv2.circle(image, pt1, 2, (0,0,255), -1)
            # cv2.circle(image, pt2, 2, (0,0,255), -1)
            # cv2.circle(image, pt3, 2, (0,0,255), -1)

            #cv2.imwrite("res.png", img)

        # # show the frame
        # cv2.imshow("Frame", frame)

        # key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("outpy.avi", fourcc, 30,
                (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")
            right_mark = cv2.imread(request.POST['right_wink_image'], -1)
            right_mark_alpha_s = right_mark[:, :, 3] / 255.0
            right_mark_alpha_l = 1.0 - right_mark_alpha_s
            
            left_mark = cv2.imread(request.POST['left_wink_image'], -1)
            left_mark_alpha_s = left_mark[:, :, 3] / 255.0
            left_mark_alpha_l = 1.0 - left_mark_alpha_s
            
            middle_mark = cv2.imread(request.POST['blink_image'], -1)
            middle_mark_alpha_s = middle_mark[:, :, 3] / 255.0
            middle_mark_alpha_l = 1.0 - middle_mark_alpha_s

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

    # cv2.imwrite('Frame', frame)
    # response = FileResponse(open('Frame', 'rb'))
    # os.remove('Frame')
    # return response




# def _grab_video(path=None, stream=None, url=None):
# 	# if the path is not None, then load the image from disk
#     if path is not None:
#         video = cv2.imread(path)
#         print(video)
# 	# otherwise, the image does not reside on disk
#     else:	
# 		# if the URL is not None, then download the image
# 		# the older ver was using resp = urllib.urlopen(url) but I updated it to resp = urllib.request.urlopen(url)
#         if url is not None:
#             resp = urllib.request.urlopen(url)
#             data = resp.read()
#         # if the stream is not None, then the image has been uploaded
#         elif stream is not None:
#             data = stream.read()
# 		# convert the image to a NumPy array and then read it into
# 		# OpenCV format
#         video = np.asarray(bytearray(data), dtype="uint8")
#         video = cv2.imdecode(video, cv2.IMREAD_COLOR)

# 	# return the image
#     print(video)
#     return video





