import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript
################################################
wCam, hCam = 640, 480
################################################
cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
vol = 0
volbar = 400

detector = htm.handDetector(detectionCon=0.7)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = math.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [50, 250], [0, 100])
        volbar = np.interp(length, [50, 250], [400, 150])

        osascript.osascript(f"set volume output volume {round(vol)}")
        print(f"set volume output volume {round(vol)}")

        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {str(int(fps))}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, round(volbar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{str(round(vol))}%', (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)