import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
print("Press Q to Exit.")
while True:
    # initialize the camera and reference picture
    frame = cap.read()[1]    
    ref= cv2.imread('ref2.jpg')

    #For use with thermal camera
    #May require new reference image
    #Human body heat
    red=255
    green=100
    blue=40
    
    # define the list of boundaries
    #[([LowerB, LowerG, LowerR], [UpperB, UpperG, UpperR])]
    #Optimal thermal boundaries _lower = [0, 0, 30] _upper = [80, 255, 255]
    # define the list of boundaries
    lower = np.array([(0), (0), (red-125)], dtype=np.uint8)
    upper = np.array([blue, green, red], dtype=np.uint8)
 
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(frame, lower, upper)
    select = cv2.bitwise_and(frame, frame, mask = mask)

    #Initialize orb method
    orb = cv2.ORB_create()

    #Compare and calculate feature distance
    kp1, des1 = orb.detectAndCompute(select,None)
    kp2, des2 = orb.detectAndCompute(ref,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Store Matches
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    #Store Matches Plot for Display
    output = cv2.drawMatches(select,kp1,ref,kp2,matches[:20],None, flags=2)
    cv2.putText(output,str(int((len(matches)/500)*100)),(350,455), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(output,"Match %:",(55,455), font, 2,(255,255,255),2,cv2.LINE_AA)
    #print(str(matches))
    
    #Display
    cv2.imshow("Vision System",output)
    
    #Exit checker
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Closed with no errors.")
