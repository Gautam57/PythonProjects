import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # the below code is a formality to get the hand module
        self.mpHands = mp.solutions.hands
        # we are going to use the same default arguments in the Hands module for now
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        # for getting all the 21 points in you hands plotted
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # has the hands module only uses the RGB colors we need to convert the image into RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # the below code checks for the hand first and for every hand present it draws the connections and dots
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # this is used to get the id number and landmark access
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # need to convert the ratio into pixel we can do it by multiplying with shape values
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList



def main():
    # to set which camera are we going to record where 0 states the camera number
    cap = cv2.VideoCapture(0)

    # Setting previous time and current time to zero
    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        # we are getting the image and status of the video that is captured
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # to display the fps int the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # imshow shows the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == '__main__':
    main()