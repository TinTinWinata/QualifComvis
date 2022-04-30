import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
from datetime import datetime
import os

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
vid = cv.VideoCapture(0)

saved_image = None


# TRAIN
train_path = 'train'
train_dir = os.listdir(train_path)
face_list = []
class_list = []
name_list = []

for i, train_dir in enumerate(train_dir):
    name_list.append(train_dir)
    for image_path in os.listdir(f'{train_path}/{train_dir}'):
        path = f'{train_path}/{train_dir}/{image_path}'
        gray = cv.imread(path, 0)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) < 1:
            continue
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = gray[y: y + w, x: x + h]
            face_list.append(face_image)
            class_list.append(i)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

while(True):
    # read video
    ret, frame = vid.read()
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
    percentage = 0
    # if there face create rectangle
    for face_rect in faces:
        x, y, w, h = face_rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        face_image = gray[y: y + w, x: x + h]
        resIdx, percentage = face_recognizer.predict(face_image)
        text = f'{name_list[resIdx]} {str(int(percentage))}%'
        cv.putText(frame, text, (x, y - 10),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

    # show frame
    cv.imshow('frame', frame)

    # if 'q' quit
    if (cv.waitKey(1) & 0xFF == ord(' ')):
        date_now = datetime.now()
        foldername = date_now.strftime("image_%d-%m-%Y-%H%M%S.jpg")
        path = 'images/' + foldername
        os.makedirs(path)
        filename = "image.jpg"
        cv.imwrite(os.path.join(path, filename), frame)
        vid.release()
        saved_image = cv.imread(path + '/' + filename, cv.IMREAD_ANYCOLOR)
        break


def showResult(nrow, ncol, res_stack):
    plt.figure(figsize=(12, 12))
    for idx, (lbl, img) in enumerate(res_stack):
        plt.subplot(nrow, ncol, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(lbl)
        plt.axis('off')
        print(path, filename)
        plt.savefig(os.path.join(path, filename))
    plt.show()


saved_image_gray = cv.cvtColor(saved_image, cv.COLOR_BGR2GRAY)

# blur
blur = cv.blur(saved_image, (10, 10))
converted_blur = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
# showResult(converted_blur, "blurried")

# canny (edge processing)
canny_050100 = cv.Canny(saved_image_gray, 50, 100)
# showResult(canny_050100, "Cannied")


# clahe
clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
cequ_gray = clahe.apply(saved_image_gray)

# threshold the clahe
_, threshold = cv.threshold(cequ_gray, 127, 255, cv.THRESH_BINARY)

# showResult(threshold, "Threshold")

# showResult(threshold, "Threshold")

# shape detector (from threshold)
shape = saved_image.copy()

_, contours, _ = cv.findContours(
    threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
)


print(contours)
i = 0


for contour in contours:
    if i == 0:
        i = 1
        continue

    if i == 30:
        break

    approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    cv.drawContours(shape, [contour], 0, (0, 0, 0), 3)
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
    i += 1


# shape detector
converted_shape = cv.cvtColor(shape, cv.COLOR_BGR2RGB)

labels = ['Blur', 'Canny', 'Threshold', 'Shape']
images = [converted_blur, canny_050100, threshold, converted_shape]

showResult(2, 2, zip(labels, images))


cv.destroyAllWindows()
