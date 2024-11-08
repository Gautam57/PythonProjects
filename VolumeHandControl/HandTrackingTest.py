import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# to set which camera are we going to record where 0 states the camera number
cap = cv2.VideoCapture(0)

# Setting previous time and current time to zero
pTime = 0
cTime = 0

detector = htm.handDetector()

while True:
    # we are getting the image and status of the video that is captured
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[2])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # to display the fps int the screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # imshow shows the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)