# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 05:39:59 2017

@author: Sunny
"""

import cv
import cv2
import math
import numpy as np
import os

# GLOBAL VARIABLES
#####################################
# Holds the pupil's center
centroid = (0,0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
#####################################


# Returns a different image filename on each call. If there are no more
# elements in the list of images, the function resets.
#
# @param list		List of images (filename)
# @return string	Next image (filename). Starts over when there are
#			no more elements in the list.
def getNewEye(list):
    global currentEye
    newEye=list[currentEye]
    currentEye+=1
    if(currentEye>=len(list)):
        currentEye=-1
    return (newEye)

# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil
def getIris(frame): 
    global radius
    iris = []
    copyImg = cv.CloneImage(frame)
    resImg = cv.CloneImage(frame)
    cv.Circle(resImg,centroid,int(radius),cv.CV_RGB(255,0,0),1)  
    grayImg = cv.CreateImage(cv.GetSize(frame), 8, 1)
    mask = cv.CreateImage(cv.GetSize(frame), 8, 1) 
    storage = cv.CreateMat(frame.width, 1, cv.CV_32FC3)
    cv.CvtColor(frame,grayImg,cv.CV_BGR2GRAY)
    cv.Canny(grayImg, grayImg, 5, 70, 3)
    cv.Smooth(grayImg,grayImg,cv.CV_GAUSSIAN, 7, 7)
    circles = getCircles(grayImg)
    iris.append(resImg)
    circle=circles[0]
    rad=circle[0][2]   
    radius = rad
    cv.Circle(resImg,centroid,rad,cv.CV_RGB(255,0,0),1)      
    return (resImg)

def segmentIris(img):
    #img=cv2.imread(direct)
    h,w,d=img.shape    
    circle_img=np.zeros((h,w),np.uint8)
    cv2.circle(circle_img,centroid,radius,255,thickness=-1)
    segmented_iris = cv2.bitwise_and(img,img,mask=circle_img)
    return segmented_iris
    

# Search middle to big circles using the Hough Transform function
# and loop for testing values in the range [80,150]. When a circle is found,
# it returns a list with the circles' data structure. Otherwise, returns an empty list.

# @param image
# @returns list
def getCircles(image):
    i = radius
    while i < 151:
        storage = cv.CreateMat(image.width, 1, cv.CV_32FC3)
        cv.HoughCircles(image, storage, cv.CV_HOUGH_GRADIENT, 2, 100.0, 30, i, 100, 140)
        circles = np.asarray(storage)
        if (len(circles) == 1):
            return circles
        i +=1
    return ([])

# Returns the same images with the pupil masked black and set the global
# variable centroid according to calculations. It uses the FindContours 
# function for finding the pupil, given a range of black tones.

# @param image		Original image for testing
# @returns image	Image with black-painted pupil
def getPupil(frame):
    pupilImg = cv.CreateImage(cv.GetSize(frame), 8, 1)
    cv.InRangeS(frame, (30,30,30), (100,100,100), pupilImg)
    contours = cv.FindContours(pupilImg, cv.CreateMemStorage(0), mode = cv.CV_RETR_EXTERNAL)
    del pupilImg
    pupilImg = cv.CloneImage(frame)
    while contours:
        moments = cv.Moments(contours)
        area = cv.GetCentralMoment(moments,0,0)
        if (area > 50):
            pupilArea = area
            x = cv.GetSpatialMoment(moments,1,0)/area
            y = cv.GetSpatialMoment(moments,0,1)/area
            pupil = contours
            global centroid
            centroid = (int(x),int(y))
            global radius
            radius = math.sqrt(area/3.141)            
            cv.DrawContours(pupilImg, pupil, (0,0,0), (0,0,0), 2, cv.CV_FILLED)
            break
        contours = contours.h_next()
    return (pupilImg)

# Returns the image as a "tape" converting polar coord. to Cartesian coord.
#
# @param image		Image with iris and pupil
# @returns image	"Normalized" image
def getPolar2CartImg(image, rad):
    imgSize = cv.GetSize(image)
    #c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
    c=centroid
    rad=int(min(float(imgSize[0]/2.0), float(imgSize[1]/2.0))) 
    imgRes = cv.CreateImage((rad*3, int(360)), 8, 3)
    #cv.LogPolar(image,imgRes,c,50.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
    cv.LogPolar(image,imgRes,c,60.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
    return (imgRes)

def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    return image

def processImg(path,savePath):
    frame = cv.LoadImage(path)   
    iris = cv.CloneImage(frame)
    output = getPupil(frame)
    radius_pupil=radius    
    iris_localized = getIris(output)
    cv.SaveImage(savePath,output)
    iris2=cv2.imread(savePath)
    segmented_iris = segmentIris(iris2)
    cv2.imwrite(savePath,segmented_iris)
    iris=cv.LoadImage(savePath)    
    normImg = getPolar2CartImg(iris,radius)
    cv.SaveImage(savePath,normImg)
    
    img=cv2.imread(savePath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    crop = autocrop(gray,30)
    crop=cv2.equalizeHist(crop)
    final=cv2.resize(crop,(60,180))
    cv2.imwrite(savePath,final)

    cv.ShowImage("input", frame)
    cv.ShowImage("localized", iris_localized)
    cv2.imshow("segmented iris",segmented_iris)
    cv.ShowImage("normalized", normImg)
    cv2.imshow("cropped",crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

ip = "S1020L01.jpg"
#dest="temp/tmp.jpg"
processImg(ip,"temp/"+ip)
#must  be present in the function calling proessimg
cv2.waitKey(0)
cv2.destroyAllWindows()
#