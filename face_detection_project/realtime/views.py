from django.shortcuts import render, HttpResponse

from pathlib import Path

# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
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

from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from wsgiref.util import FileWrapper

def try_glasses_image(request):
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
    
    image = _grab_image(stream=request.FILES['image'])
    
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    glasses_mark = cv2.imread(request.POST['glasses_image'], -1)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        left_midpoint = eye_midpoint(leftEye)
        right_midpoint = eye_midpoint(rightEye)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
    
        lEdge = (lEnd + lStart) // 2
        distance = shape[lEdge][0] - shape[rStart][0]
        #dim = (width, height)
        resized_glasses = imutils.resize(glasses_mark, width=int(distance * 1.8))

        horizontal_shift = int(-distance * 0.45)
        vertical_shift = int(-distance * 0.2)
        # step 2: find starting point based on lStart
        y1, y2 = shape[rStart][1] + vertical_shift, shape[rStart][1] + resized_glasses.shape[0] + vertical_shift
        x1, x2 = shape[rStart][0] + horizontal_shift, shape[rStart][0] + resized_glasses.shape[1] + horizontal_shift

        # need to account for mark going out of image
        y1_offset = max(y1, 0) - y1
        x1_offset = max(x1, 0) - x1
        y2_offset = y2 - min(y2, image.shape[0])
        x2_offset = x2 - min(x2, image.shape[1])
    
        glasses_mark_alpha_s = resized_glasses[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, 3] / 255.0
        glasses_mark_alpha_l = 1.0 - glasses_mark_alpha_s

        for c in range(0, 3):
            image[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c] = (
                glasses_mark_alpha_s * resized_glasses[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, c] +
                glasses_mark_alpha_l * image[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c]
                )

    cv2.imwrite('temp.jpg', image)
    response = FileResponse(open('temp.jpg', 'rb'))
    os.remove('temp.jpg')
    return response


def eye_midpoint(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	# A = dist.euclidean(eye[1], eye[5])
	# B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	midpoint = C/2.0 

	# return the eye aspect ratio
	return midpoint

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
def try_glasses_video(request):
    # EYE_AR_THRESH = 0.3
    # EYE_AR_CONSEC_FRAMES = 3

    # # initialize the frame counters and the total number of blinks
    # COUNTER = 0
    # TOTAL = 0

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
    print("this is test", data)
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
    glasses_mark = None
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
        
        # check if the writer is None
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            #get the dimention of the elements up to [2], so get [0] and [1]
            #and make a new viode with 30 frames/sec
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter("outpy.avi", fourcc, 30,
                (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")
            glasses_mark = cv2.imread(request.POST['glasses_image'], -1)
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
            left_midpoint = eye_midpoint(leftEye)
            right_midpoint = eye_midpoint(rightEye)

            # average the eye aspect ratio together for both eyes
            # ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(frame, [leftEyeHull], -1, (20, 20, 255), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (20, 20, 255), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            # if ear < EYE_AR_THRESH:
            #     COUNTER += 1

            # # otherwise, the eye aspect ratio is not below the blink
            # # threshold
            # else:
            #     # if the eyes were closed for a sufficient number of
            #     # then increment the total number of blinks
            #     if COUNTER >= EYE_AR_CONSEC_FRAMES:
            #         TOTAL += 1
            #     # set star countdown variable to 10 frames

            #     # reset the eye frame counter
            #     COUNTER = 0

            # # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # y1, y2 = 100, 100 + glasses_mark.shape[0]
            # x1, x2 = 50, 50 + glasses_mark.shape[1]
            
            # the eye position and distances change every frame, so need to compute glasses here!
            # step 1: resize glasses
            # width_scale_percent = abs((shape[(lEnd-lStart)/2][0] - shape[lStart][0]) -shape[glasses_mark][0])/shape[(lEnd-lStart)/2][0]*100
            # width = shape[(lEnd-lStart)/2][0] - shape[lStart][0]
            # height = int(glasses_mark.shape[1] * width_scale_percent / 100)
            lEdge = (lEnd + lStart) // 2
            distance = shape[lEdge][0] - shape[rStart][0]
            #dim = (width, height)
            resized_glasses = imutils.resize(glasses_mark, width=int(distance * 1.8))

            horizontal_shift = int(-distance * 0.45)
            vertical_shift = int(-distance * 0.2)
            # step 2: find starting point based on lStart
            y1, y2 = shape[rStart][1] + vertical_shift, shape[rStart][1] + resized_glasses.shape[0] + vertical_shift
            x1, x2 = shape[rStart][0] + horizontal_shift, shape[rStart][0] + resized_glasses.shape[1] + horizontal_shift

            # need to account for mark going out of frame
            y1_offset = max(y1, 0) - y1
            x1_offset = max(x1, 0) - x1
            y2_offset = y2 - min(y2, frame.shape[0])
            x2_offset = x2 - min(x2, frame.shape[1])
    
            glasses_mark_alpha_s = resized_glasses[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, 3] / 255.0
            # ((lEnd-lStart)/2-rStart)
            glasses_mark_alpha_l = 1.0 - glasses_mark_alpha_s

            for c in range(0, 3):
                    frame[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c] = (glasses_mark_alpha_s * resized_glasses[y1_offset:y2-y1-y2_offset, x1_offset:x2-x1-x2_offset, c] +
                                              glasses_mark_alpha_l * frame[y1+y1_offset:y2-y2_offset, x1+x1_offset:x2-x2_offset, c])
        # ((lEnd-lStart)/2-rStart)
        # # show the frame
        # cv2.imshow("Frame", frame)

        # key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

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


