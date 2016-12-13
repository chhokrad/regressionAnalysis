import numpy as np
import cv2
import time
img = cv2.imread('pic.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray,None)
end =time.time()
