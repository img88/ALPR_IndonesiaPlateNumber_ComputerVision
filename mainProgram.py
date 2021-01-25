import cv2 as cv
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

#------------------------------------------------------
# Preproccessing
#------------------------------------------------------

img = cv.imread(r'E:\test images\AA5627JT.jpg') # load an image
img = cv.resize(img, (int(img.shape[1]*.4),int(img.shape[0]*.4))) # resizing the image
im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert bgr to grayscale

# Vehicle images have different illumination level, let's normalize it

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20)) # create kernel/structuring element, shape = ellipse, diameter = 20 pixel
im_open = cv.morphologyEx(im_gray, cv.MORPH_OPEN, kernel) # apply morph open with kernel above, opening = erode->dilate
im_subs = im_gray - im_open # substract those images
(thresh, im_bw) = cv.threshold(im_subs, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # convert from grayscale to black and white using otsu's thresholding

#------------------------------------------------------
# Plate Detection
#------------------------------------------------------

# Plate detection using contours

contours, hierarchy = cv.findContours(im_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get the contour for every area
index = 0 # counter index of cnt in contours
idxs = [] # index of cnt in contours that contains plate location candidate 

for cnt in contours:
    x,y,w,h = cv.boundingRect(cnt) # get the x, y, width, height values from cnt in contours
    aspect_ratio = w/h # calculate the aspect ratio
    # get the plate location candidate if pixel width more than or equal to 200 pixels and aspect ratio less than or equal to 4
    if w >= 200 and aspect_ratio <= 4 : 
        idxs.append(index) # get the index value of contours that contain plate location candidate
    index += 1

# Get plate number location
#   in my experiment, we can get one or two plate locations
#   if we get one, that is the plate number
#   if we get two, the second location is the plate number with a smaller size
#   the plate number will be stored to im_plate by cropping im_gray

for a in idxs:
    xp,yp,wp,hp = cv.boundingRect(contours[a])

if len(idxs) == 1: 
    cv.rectangle(img,(xp,yp),(xp+wp,yp+hp),(0,255,0),5)
    im_plate = im_gray[yp:yp+hp, xp:xp+wp]
else:
    print(' more than one plate detected, save the second box')
    xp,yp,wp,hp = cv.boundingRect(contours[index[1]])
    cv.rectangle(img,(xp,yp),(xp+wp,yp+hp),(0,255,0),5)
    im_plate = im_gray[yp:yp+hp, xp:xp+wp]

#------------------------------------------------------
# Character Segmentation
#------------------------------------------------------

(thresh, im_bw) = cv.threshold(im_plate, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # convert from grayscale to black and white
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3)) # create kernel, shape = cross, size 3,3
im_bw = cv.morphologyEx(im_bw, cv.MORPH_OPEN, kernel) # apply morph open

# Character segmentation using contours

contours, hierarchy = cv.findContours(im_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
idx = 0 # counter index of contour in contours
index = [] # the character candidate will be stored here
im_plate2 = im_plate.copy() # copy im_plate

# Get the character candidate
#   if (the height in contours more than or equal to 40 pixels) and (less than or equal to 60 pixels)
#   and if (the width more than or equal to 10 pixels)

for contour in contours:
    x,y,w,h = cv.boundingRect(contour)
    if (h >= 40 and h <= 60) and (w >=10):
        index.append(idx)
    idx += 1

# if there is no char candidate then print
if index == []:
    print('Chars not detected')
else:

    # Get the real character
    #   the license plate numbers (character) have the same height in y axis.
    #   so, compare the height of each candidate with the other candidates, 
    #   if the difference is not more than 11 pixels then give 1 point to the score.
    #   the real character will have the highest and the same score

    # Scoring
    score = np.zeros(len(index))
    c = 0
    for a in index:
        x1,y1,w1,h1 = cv.boundingRect(contours[a])
        for b in index:
            if a == b:
                continue
            else:
                x2,y2,w2,h2 = cv.boundingRect(contours[b])
                ydiff = abs(y1 - y2)
                if ydiff < 11:
                    score[c] = score[c] + 1 
        c += 1

    # Get the highest and the same score
    #   character will be stored in chars

    chars = []
    b = 0
    for a in score:
        if a == max(score):
            chars.append(index[b])
        b += 1

    # Here, we've found the character, but..
    # the arrangement is still random, for example: Z 1234 AB, the character stored: 1 3Z24 BA
    # we have to sort those character by using x value
    # sorted character will be stored to real_chars

    a = 0
    xcoors = []
    for char in chars:
        x, y, w, h = cv.boundingRect(contours[char])
        xcoors.append(x) # get the x value

    xcoors = sorted(xcoors) # sort the x value (small --> large)

    real_chars = []
    for xcoor in xcoors:
        for char in chars:
            x, y, w, h = cv.boundingRect(contours[char])
            if xcoors[xcoors.index(xcoor)] == x:
                real_chars.append(char) # storing sorted character

    im_plate = cv.cvtColor(im_plate, cv.COLOR_GRAY2BGR)

    # Draw rectangle for each character
    for rc in real_chars:
        x,y,w,h = cv.boundingRect(contours[rc])
        cv.rectangle(im_plate,(x,y),(x+w,y+h),(0,255,0),5)
        cv.imshow('Character', im_plate) # show character in plate number

#------------------------------------------------------
# Character classification
#------------------------------------------------------

    # i follow this instruction for image classification: 
    # https://www.tensorflow.org/tutorials/images/classification
    # the classification result will be stored to plate_number

    img_height = 40 # image height
    img_width = 40 # image width

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    model = keras.models.load_model('my_model') # load model

    num_plate = []

    for rc in real_chars:
        x,y,w,h = cv.boundingRect(contours[rc])
        char_crop = cv.cvtColor(im_bw[y:y+h,x:x+w], cv.COLOR_GRAY2BGR)

        char_crop = cv.resize(char_crop, (img_width, img_height))

        img_array = keras.preprocessing.image.img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array) # make predictions
        score = tf.nn.softmax(predictions[0]) 

        num_plate.append(class_names[np.argmax(score)])
        print(class_names[np.argmax(score)], end='')

    # Show the result

    plate_number = ''
    for a in num_plate:
        plate_number += a

    cv.putText(img, plate_number,(xp, yp + hp + 50), cv.FONT_ITALIC, 2.0, (0,255,0), 3)
    cv.imshow(plate_number, img)
    cv.waitKey(0)
