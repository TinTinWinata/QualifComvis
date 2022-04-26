import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv.VideoCapture(0)

last_frame = None


def showResult(last_frame, lbl):
    plt.figure(figsize=(12, 12))
    plt.imshow(last_frame, cmap='gray')
    plt.title(lbl)
    plt.axis('off')
    plt.show()


while(True):
    # read video
    ret, frame = vid.read()
    last_frame = frame
    # get gray color
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Image Processing (clahe contrast)
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cequ_gray = clahe.apply(gray)

    # get face (canny)
    faces = faceCascade.detectMultiScale(
        cequ_gray,
        scaleFactor=1.2,
        minNeighbors=5,
    )

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

# blur
blur = cv.blur(last_frame, (10, 10))
showResult(blur, "blurried")

# canny (edge processing)
canny_050100 = cv.Canny(last_frame_gray, 50, 100)
showResult(canny_050100, "Cannied")

cv.destroyAllWindows()


# shape
shape = last_frame.copy()

_, threshold = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
_, contours, _ = cv.findContours(
    threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
i = 0
for contour in contours:
    if i == 0:
        i = 1
        continue
approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
cv.drawContours(shape, [contour], 0, (0, 0, 255), 5)
M = cv.moments(contour)
if M['m00'] != 0.0:
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    if len(approx) == 3:
        cv.putText(shape, 'Triangle', (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 4:
        cv.putText(shape, 'Quadrilateral', (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 5:
        cv.putText(shape, 'Pentagon', (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif len(approx) == 6:
        cv.putText(shape, 'Hexagon', (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else:
        cv.putText(shape, 'circle', (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

showResult(shape, 'shape')
