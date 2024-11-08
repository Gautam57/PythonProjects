import cv2
import mediapipe as mp
import time

# to set which camera are we going to record where 0 states the camera number
cap = cv2.VideoCapture(0)

# Setting previous time and current time to zero
pTime = 0
cTime = 0
# the below code is a formality to get the hand module
mpHands = mp.solutions.hands
# we are going to use the same default arguments in the Hands module for now
hands = mpHands.Hands()
# for getting all the 21 points in you hands plotted
mpDraw = mp.solutions.drawing_utils

while True:
    # we are getting the image and status of the video that is captured
    success, img = cap.read()
    # has the hands module only uses the RGB colors we need to convert the image into RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # the below code checks for the hand first and for every hand present it draws the connections and dots
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # this is used to get the id number and landmark access
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                # need to convert the ratio into pixel we can do it by multiplying with shape values
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                print(id, cx, cy)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # to display the fps int the screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # imshow shows the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)