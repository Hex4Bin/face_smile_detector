from cv2 import cv2
from random import randrange


# Our pre-trained face classifier
CLASSIFIER_FACE = './haarcascade_frontalface_default.xml'

# load some pre-trained frontal face data from opencv
face_tracker = cv2.CascadeClassifier(CLASSIFIER_FACE)

# load an image and convert it to grayscaled version for test
img = cv2.imread('2faces.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = face_tracker.detectMultiScale(
    image=grayscaled_img, scaleFactor=1.2, minNeighbors=6, minSize=(120, 120))

# the coordinates come this format: [[100 115 225 225]]
# we have to make a tuple based on the data
# draw rectangles around the faces with some randomly generated colors
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h),
                  (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

# display the image
cv2.imshow('Face Detector', img)
cv2.waitKey()
cv2.destroyAllWindows()
