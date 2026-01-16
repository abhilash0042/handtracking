import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Custom Premium Drawing: White Lines, Yellow Joints
                    # Draw connections (White Lines)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),
                                               self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
                                               )
                    # Draw joints explicitly as Yellow Circles for that "Cyber" look
                    h, w, c = img.shape
                    for id, lm in enumerate(handLms.landmark):
                         cx, cy = int(lm.x * w), int(lm.y * h)
                         cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED) # Yellow Fill
                         cv2.circle(img, (cx, cy), 6, (0, 0, 0), 1) # Black outline

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    if id == 8: # Index Tip Highlight
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []
            
        # Thumb
        # Determine Left/Right hand for thumb logic.
        # For this simple version, assume standard right hand behavior or check x relative to v.
        # Tip x < IP x  (for Right Hand)
        if self.lmList[self.tipIds[0]][0] < self.lmList[self.tipIds[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
