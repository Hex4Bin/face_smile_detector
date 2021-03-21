from cv2 import cv2
from random import randrange


# Our pre-trained face classifier
classifier_face = './haarcascade_frontalface_default.xml'

# load some pre-trained frontal face data from opencv
face_tracker = cv2.CascadeClassifier(classifier_face)

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
    face_coordinates = face_tracker.detectMultiScale(
        image=grayscaled_video, scaleFactor=1.2, minNeighbors=6, minSize=(120, 120))

    # the coordinates come this format: [[100 115 225 225]]
    # we have to make a tuple based on the data
    # draw rectangles around the faces with some randomly generated colors
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

    cv2.imshow('PRESS ESC TO QUIT', frame)
    key = cv2.waitKey(1)

    # stop if "ESC" is pressed
    if key == 27:
        break

# release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()
