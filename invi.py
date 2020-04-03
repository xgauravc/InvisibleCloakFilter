import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0) # access video from webcam

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

time.sleep(3)
count = 0
background = 0

for i in range(60):
    ret, background = cap.read()

print('background selected')

while(cap.isOpened()):
    ret, img = cap.read()  # video frames getting stored in img
    if not ret:
        break
    count += 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # converting img to hsv

    lower_red = np.array([0, 120, 70]) # range of color
    upper_red = np.array([10,255,255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow('Mask1', mask1) # show video
    

    lower_red = np.array([170,120,70])
    uper_red = np.array([180,255,255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow('Mask2', mask2) # show video

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=2)
##    cv2.imshow('MorpMask', mask1)
    mask1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)
    cv2.imshow('notMask',mask2)

    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imshow('Res', res2)
    final_output = cv2.addWeighted(res1, 1, res2, 1,0)
    output.write(final_output)

    cv2.imshow('Imgee', final_output)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()



    
    
