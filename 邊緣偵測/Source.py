# -*- coding: utf-8 -*-
import cv2
from skimage.filters import sobel

image1 = cv2.imread('plane.jpg')                 #讀取圖片
image2 = cv2.imread('insect.png')                #讀取圖片

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #獲得灰階圖片
eq1 = cv2.equalizeHist(gray1)                    #獲得均值灰階圖片

cv2.imshow("Gray1", gray1)                       #輸出灰階圖片
cv2.imshow("Histogram Equalization1", eq1)       #輸出均值灰階圖片
sobel1 = sobel(gray1)                            #獲得邊界圖片
cv2.imshow("Sobel operator1",sobel1)             #輸出邊界圖片
##################################################以下同上
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
eq2 = cv2.equalizeHist(gray2)

cv2.imshow("Gray2", gray2)
cv2.imshow("Histogram Equalization2", eq2)
sobel2 = sobel(gray2)
cv2.imshow("Sobel operator2",sobel2)
##################################################
cv2.waitKey()                                    #防止畫面卡死
cv2.destroyAllWindows()                          #防止畫面卡死