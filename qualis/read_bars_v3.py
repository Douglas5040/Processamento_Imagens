import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imgs/teste1.jpeg')
blur_hor = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
blur_vert = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
mask = ((img[:,:,0]>blur_hor*1.2) | (img[:,:,0]>blur_vert*1.2)).astype(np.uint8)*255

ret,thresh = cv2.threshold(mask,127,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)

count = 0

print('contours',len(contours))
for i in range(len(contours)):

  #count = count+1
  x,y,w,h = cv2.boundingRect(contours[i]) 
  rect = cv2.minAreaRect(contours[i])
  area = cv2.contourArea(contours[i])
  box = cv2.boxPoints(rect)
  ratio = w/h
  M = cv2.moments(contours[i])

  #print(x,y,w,h)
  #print(rect)
  #print(area)
  # print(ratio)
  # print(hierarchy[0][i][2])
  # print(box)
  # print('------\n')

  if M["m00"] == 0.0:
         cX = int(M["m10"] / 1 )
         cY = int(M["m01"] / 1 )

  if M["m00"] != 0.0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
  
  img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,0, 255), 1)
  mask = cv2.rectangle(mask, (x,y), (x+w,y+h), (0,0, 255), 1)

  if (area > 23600 and area < 26300 and hierarchy[0][i][2] < 0 and (ratio > 12 and ratio < 14)):
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
    
    #cv2.circle(img, (cX, cY), 1, (255, 255, 255), -1)
    count = count + 1 



print(count)

cv2.imshow("m - Mask",mask)
cv2.imshow("f - Original",img)
cv2.waitKey(0)