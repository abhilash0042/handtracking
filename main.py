import cv2
import numpy as np
import mediapipe as mp
import math
import collections

# ==========================================
# 1. HEURISTIC SHAPE VALIDATOR
# ==========================================
class ShapeValidator:
    def __init__(self):
        self.ref_size = (100, 100)

    def generate_reference_shape(self, char):
        """Generates a binary mask of the target char for shape comparison."""
        img = np.zeros(self.ref_size, dtype=np.uint8)
        # Center the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3
        thickness = 5
        (w, h), _ = cv2.getTextSize(char, font, scale, thickness)
        x = (self.ref_size[0] - w) // 2
        y = (self.ref_size[1] + h) // 2
        cv2.putText(img, char, (x, y), font, scale, 255, thickness)
        return img

    def validate(self, drawing_img, target_char, stroke_count):
        """
        Validates the drawing against the target character using heuristics:
        1. Shape Matching (Hu Moments)
        2. Aspect Ratio (Bounding Box)
        3. Stroke Count (Loose check)
        """
        if drawing_img is None: return 0.0, "Empty"

        # 1. Preprocess Drawing
        gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None: return 0.0, "Empty"
        
        # Bounding Box
        x, y, w, h = cv2.boundingRect(coords)
        if w < 10 or h < 10: return 0.0, "Too Small"
        
        # Extract ROI and Resize
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, self.ref_size)
        _, roi_bin = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)

        # 2. Get Reference
        ref_img = self.generate_reference_shape(target_char)
        
        # 3. Shape Matching (Hu Moments)
        # matchShapes returns a metric where lower is better (0 = perfect match)
        # We define score = 1 / (1 + distance)
        dist = cv2.matchShapes(roi_bin, ref_img, cv2.CONTOURS_MATCH_I1, 0)
        shape_score = 1.0 / (1.0 + dist) # 0 to 1

        # 4. Aspect Ratio Check
        user_ratio = w / h
        # Get ref ratio
        ref_coords = cv2.findNonZero(ref_img)
        rx, ry, rw, rh = cv2.boundingRect(ref_coords)
        ref_ratio = rw / rh
        
        ratio_diff = abs(user_ratio - ref_ratio)
        ratio_score = max(0, 1.0 - ratio_diff) # Simple linear penalty
        
        # 5. Combined Score
        # Weight shape more than ratio
        final_score = (shape_score * 0.7) + (ratio_score * 0.3)
        
        # Debug info
        details = f"Shape:{shape_score:.2f} Ratio:{ratio_score:.2f} Strokes:{stroke_count}"
        
        return final_score, details

# ==========================================
# 2. ALPHABET WHEEL UI
# ==========================================
class AlphabetWheel:
    def __init__(self, pos=(150, 150), radius=100):
        self.cx, self.cy = pos
        self.radius = radius
        self.chars = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        self.angle_step = 360 / len(self.chars)
        self.selected_idx = 0
        self.current_angle = 0 # For visualization

    def update(self, hand_x, hand_y):
        """Rotates wheel based on hand position relative to center"""
        if hand_x == 0 and hand_y == 0: return

        # Calculate angle
        dx = hand_x - self.cx
        dy = hand_y - self.cy
        # atan2 returns radians from -pi to pi.
        # We want 0-360 starting from top (270 degrees in standard math is top, or -90)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360
        if angle_deg < 0: angle_deg += 360
        
        # Visual rotation logic:
        # We want the "Selection Zone" to be at the TOP (270 deg).
        # We rotate the wheel so that the section corresponding to the hand angle aligns with the top.
        # OR simpler: The hand points to the letter.
        # Let's start with Hand Points to Letter.
        
        # Adjust 0 to be Top for intuition?
        # Standard: 0 is Right, 90 Down, 180 Left, 270 Top.
        # Let's shift so 0 is Top ( subtract 270 or add 90)
        adjusted_angle = (angle_deg + 90) % 360
        
        # Index
        self.selected_idx = int(adjusted_angle / self.angle_step) % 26
        self.current_angle = angle_deg

    def get_selected_char(self):
        return self.chars[self.selected_idx]

    def draw(self, img, active=True):
        overlay = img.copy()
        
        # Thicker Wheel Background
        cv2.circle(overlay, (self.cx, self.cy), self.radius + 15, (30, 30, 30), cv2.FILLED)
        cv2.circle(overlay, (self.cx, self.cy), self.radius + 15, (255, 255, 255), 3) # White outer
        cv2.circle(overlay, (self.cx, self.cy), self.radius - 35, (0, 0, 0), cv2.FILLED) # Hollow center

        # Alpha Blend for Transparent look
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Draw Chars
        for i, char in enumerate(self.chars):
            # Calculate position
            # Start from Top (-90 degrees)
            theta = math.radians(i * self.angle_step - 90)
            
            x_pos = int(self.cx + (self.radius - 10) * math.cos(theta))
            y_pos = int(self.cy + (self.radius - 10) * math.sin(theta))
            
            # Selection Highlight
            if i == self.selected_idx:
                # Glowing Yellow Bubble
                cv2.circle(img, (x_pos, y_pos), 22, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x_pos, y_pos), 24, (255, 255, 255), 2)
                # Black Text
                cv2.putText(img, char, (x_pos-10, y_pos+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            else:
                # White Text
                color = (200, 200, 200)
                cv2.putText(img, char, (x_pos-8, y_pos+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ==========================================
# 3. MAIN APP STRUCTURE
# ==========================================
import time

class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width, self.height = 1280, 720
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        # Components
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.wheel = AlphabetWheel(pos=(200, 200), radius=120)
        self.validator = ShapeValidator()
        
        # State
        self.sm_x, self.sm_y = 0, 0 # Smooth coords
        self.alpha = 0.3 # Smoothing factor (Lower = Smoother)
        
        self.drawing = False
        self.strokes = [] # List of finished stroke paths [ [(x,y),...], [(x,y),...] ]
        self.current_stroke = [] # Active stroke
        
        self.validation_result = ("?", 0.0, "Ready", (200, 200, 200))
        self.termination_counters = 0
        
        # Tracking Loss Handling
        self.last_hand_time = time.time()
        self.stroke_break_threshold = 0.4 # seconds

    def run(self):
        print("Starting Air Writing App...")
        while True:
            success, img = self.cap.read()
            if not success: break
            img = cv2.flip(img, 1)

            # 1. Processing (Hand Tracking)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            
            x1, y1 = 0, 0
            fingers = []
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                self.last_hand_time = time.time() # Update valid time
                
                for handLms in results.multi_hand_landmarks:
                    # Parse Landmarks
                    lmList = []
                    h, w, c = img.shape
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    
                    if lmList:
                        # Index Tip
                        x1, y1 = lmList[8][1:]
                        # Fingers Up Check
                        fingers = self.check_fingers(lmList)

            # 2. Logic Update
            
            # A. Smoothing & Coordinates
            if hand_detected and x1 != 0:
                if self.sm_x == 0: self.sm_x, self.sm_y = x1, y1
                self.sm_x = self.alpha * x1 + (1 - self.alpha) * self.sm_x
                self.sm_y = self.alpha * y1 + (1 - self.alpha) * self.sm_y
                draw_point = (int(self.sm_x), int(self.sm_y))
                
                # B. Wheel Update (Independent)
                self.wheel.update(draw_point[0], draw_point[1])
            else:
                draw_point = None

            target_char = self.wheel.get_selected_char()

            # C. Gesture Handling
            if hand_detected and fingers: 
                # TERMINATE (Index + Middle)
                if fingers[1] and fingers[2] and not fingers[3]:
                    self.termination_counters += 1
                    cv2.putText(img, f"EXITING... {3 - self.termination_counters//15}", (self.width//2-100, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
                    if self.termination_counters > 45: break
                else:
                    self.termination_counters = 0

                # DRAW (Index only)
                if fingers[1] and not fingers[2] and not fingers[3]:
                    self.drawing = True # User is intentionally drawing
                    if draw_point:
                        self.current_stroke.append(draw_point)
                
                # STOP / VALIDATE (Index Down)
                # Ensure we only stop if we were actually drawing and we explicitly lower fingers
                elif self.drawing and not fingers[1]: 
                    self.drawing = False
                    if self.current_stroke:
                        # Only finalize if it has significant length to avoid dots
                        if len(self.current_stroke) > 2:
                            self.strokes.append(list(self.current_stroke))
                            # Validate
                            # Create a temp canvas for OCR
                            temp_canvas = np.zeros((self.height, self.width, 3), np.uint8)
                            for s in self.strokes:
                                if len(s) > 1: cv2.polylines(temp_canvas, [np.array(s)], False, (255,255,255), 15)
                            
                            score, details = self.validator.validate(temp_canvas, target_char, len(self.strokes))
                            
                            match_res = "MATCH!" if score > 0.8 else "..."
                            color = (0, 255, 0) if score > 0.8 else (0, 0, 255)
                            self.validation_result = (target_char, score, match_res, color)
                            print(f"Target: {target_char} | Score: {score:.2f} | {details}")
                    self.current_stroke = [] # Clear active
                    
                # CLEAR (Open Palm)
                if all(fingers[1:]):
                     self.strokes = []
                     self.current_stroke = []
                     self.validation_result = ("?", 0.0, "Ready", (200, 200, 200))

            # D. Handle Tracking Loss (Robustness)
            else:
                # No hand detected
                # Check how long it's been
                if self.drawing and (time.time() - self.last_hand_time > self.stroke_break_threshold):
                    # Timeout exceeded, commit stroke
                    self.drawing = False
                    if self.current_stroke:
                         self.strokes.append(list(self.current_stroke))
                    self.current_stroke = []


            # 3. Rendering Pipeline
            
            # Dark Overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (self.width, self.height), (0,0,0), cv2.FILLED)
            cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
            
            # Wheel
            self.wheel.draw(img)
            
            # Landmarks
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4))

            # Strokes (Polylines for smoothness)
            # Draw Finished Strokes
            for s in self.strokes:
                if len(s) > 1:
                    # White core
                    cv2.polylines(img, [np.array(s)], False, (255, 255, 255), 15, cv2.LINE_AA)
                    # Neon Glow (Cyan) - Simple outline effect
                    cv2.polylines(img, [np.array(s)], False, (255, 255, 0), 19, cv2.LINE_AA)
            
            # Draw Current Stroke
            if len(self.current_stroke) > 1:
                cv2.polylines(img, [np.array(self.current_stroke)], False, (255, 255, 255), 15, cv2.LINE_AA)
                cv2.polylines(img, [np.array(self.current_stroke)], False, (255, 0, 255), 19, cv2.LINE_AA) # Magenta Glow for active

            
            # Text UI
            cv2.rectangle(img, (self.width-300, 0), (self.width, 150), (20, 20, 20), cv2.FILLED)
            cv2.putText(img, f"Target: {target_char}", (self.width-280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if len(self.validation_result) >= 4:
                 res_char, res_score, res_msg, res_col = self.validation_result
                 cv2.putText(img, f"{res_msg} ({int(res_score*100)}%)", (self.width-280, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, res_col, 2)
            
            cv2.imshow("Futuristic Air Writing", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def check_fingers(self, lmList):
        # ... Reuse logic ...
        fingers = []
        if lmList[4][0] < lmList[3][0]: fingers.append(1)
        else: fingers.append(0)
        tips = [8, 12, 16, 20]
        for id in tips:
            if lmList[id][2] < lmList[id-2][2]: fingers.append(1)
            else: fingers.append(0)
        return fingers

if __name__ == "__main__":
    app = App()
    app.run()
