from cv2 import cv2
from random import randrange

"""
# load some pre-trained frontal face data from opencv
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# load an image and convert it to grayscaled version for test
img = cv2.imread('2faces.jpg')
#img = cv2.imread('rdj.jpg')
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# the coordinates come this format: [[100 115 225 225]]
# we have to make a tuple based on the data
for (x, y, w, h) in face_coordinates:
    # draw rectangles around the faces with some randomly generated colors
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

# display the image
cv2.imshow('Test', img)
cv2.waitKey()
"""

# load some pre-trained frontal face data from opencv
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# capture video from webcam
# 0 means the default webcam
webcam = cv2.VideoCapture(0)

# iterate over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # convert to grayscale
    grayscaled_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_video)

    # the coordinates come this format: [[100 115 225 225]]
    # we have to make a tuple based on the data
    # draw rectangles around the faces with some randomly generated colors
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

    cv2.imshow('PRESS ESC TO QUIT', frame)
    key = cv2.waitKey(1)

    # stop if "q" is pressed
    if key == 27:
        break

# release the VideoCapture object
webcam.release()
