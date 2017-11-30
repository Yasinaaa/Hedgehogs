import Image

import cv2
from matplotlib import pyplot as plt

# Get user supplied values
imagePath = "res/the_saturdays_right.jpg"
cascPath = "res/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

source_image = Image.open(imagePath)
faces_array = []

for (x, y, w, h) in faces:
    coords = (x, y, x + w, y + h)
    curent_face = source_image.crop(coords)
    faces_array.append(curent_face)
    curent_face.show()

cv2.waitKey(0)

