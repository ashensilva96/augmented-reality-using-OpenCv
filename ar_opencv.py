# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:23:03 2022

@author: HP
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

image_target = cv2.imread("image_ref.jpg")
video_target = cv2.VideoCapture("video.mp4")

detection = False
frame_counter = 0

success, img_video = video_target.read()

iH, iW, iC = image_target.shape
img_video = cv2.resize(img_video, (iW, iH))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(image_target, None)
#image_target = cv2.drawKeypoints(image_target, kp1, None)

while True:
    success, webcam_img = cap.read()
    imgAug = webcam_img.copy()
    kp2, des2 = orb.detectAndCompute(webcam_img, None)
    #webcam_img = cv2.drawKeypoints(webcam_img, kp2, None)
    
    if detection == False:
        video_target.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
    else:
        if frame_counter == video_target.get(cv2.CAP_PROP_FRAME_COUNT):
           video_target.set(cv2.CAP_PROP_POS_FRAMES, 0)
           frame_counter = 0
        success, img_video = video_target.read()
        img_video = cv2.resize(img_video, (iW, iH))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    imageFeatures = cv2.drawMatches(
        image_target, kp1, webcam_img, kp2, good, None, flags=2)

    if len(good) > 25:
        detection = True
        srcPts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dtsPts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dtsPts, cv2.RANSAC, 5)

        print(matrix)

        pts = np.float32([[0, 0], [0, iH], [iW, iH], [iW, 0]]
                         ).reshape(-1, 1, 2)
        dts = cv2.perspectiveTransform(pts, matrix)
        img_2 = cv2.polylines(
            webcam_img, [np.int32(dts)], True, (255, 0, 255), 3)

        imgWrap = cv2.warpPerspective(
            img_video, matrix, (webcam_img.shape[1], webcam_img.shape[0]))

        maskNew = np.zeros((webcam_img.shape[0], webcam_img.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dts)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask= maskInv)
        imgAug = cv2.bitwise_or(imgWrap, imgAug)
        imgAug = cv2.putText(imgAug, "Augmented Reality - OpenCV", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        imgAug = cv2.putText(imgAug, "developer : ashensilva", (50, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        

    else:
        imgAug = cv2.putText(imgAug, "recognizing...", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        imgAug = cv2.putText(imgAug, "developer : ashensilva", (50, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("imgAug", imgAug)
    #cv2.imshow("imgWrap", imgWrap)
    #cv2.imshow("img_2", img_2)
    #cv2.imshow("imageFeatures", imageFeatures)
    #cv2.imshow("image target", image_target)
    #cv2.imshow("video_target", img_video)
    #cv2.imshow("Web_image", webcam_img)

    c = cv2.waitKey(1)
    frame_counter +=1
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
