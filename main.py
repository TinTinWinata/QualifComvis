import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv.VideoCapture(0)

last_frame = None


def showResult(img, lbl):
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray')
    plt.title(lbl)
    plt.axis('off')
    plt.show()


while(True):
    # read video
    ret, frame = vid.read()

    # get gray color
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Image Processing (clahe contrast)
    # clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # cequ_gray = clahe.apply(gray)

    # get face (canny)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
    )

    # If no face continue
    if len(faces) < 1:
        continue

    # if there face create rectangle
    for face_rect in faces:
        x, y, w, h = face_rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(frame, "People", (x, y - 10),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

    # show frame
    cv.imshow('frame', frame)

    # if 'q' quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()


# take gray at last frame
last_frame_gray = cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)

# canny (edge processing)
canny_050100 = cv.Canny(last_frame_gray, 50, 100)

showResult(canny_050100, "Cannied")


cv.destroyAllWindows()
