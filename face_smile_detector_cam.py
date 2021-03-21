from cv2 import cv2
from random import randrange


# Our pre-trained face&smile classifier
CLASSIFIER_FACE = './haarcascade_frontalface_default.xml'
CLASSIFIER_SMILE = './haarcascade_smile.xml'

# load some pre-trained face&smile data from opencv
face_tracker = cv2.CascadeClassifier(CLASSIFIER_FACE)
smile_tracker = cv2.CascadeClassifier(CLASSIFIER_SMILE)

# capture video from webcam
# 0 means the default webcam
webcam = cv2.VideoCapture(0)

# iterate over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # If there is an error, abort
    if not successful_frame_read:
        break

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

        # get the subframe (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        # convert to grayscale
        grayscaled_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detect smiles
        smile_coordinates = smile_tracker.detectMultiScale(
            image=grayscaled_face, scaleFactor=1.05, minNeighbors=500)

        """
        # find all smiles in the face
        for (x_, y_, w_, h_) in smile_coordinates:

            # draw a rectangle around the smile
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),
                          (50, 50, 200), 4)
        """

        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(50, 50, 200), thickness=5)

    # display the image
    cv2.imshow('PRESS ESC TO QUIT', frame)
    key = cv2.waitKey(1)

    # stop if "ESC" is pressed
    if key == 27:
        break

# release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()
