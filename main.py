import cv2
import numpy as np
import mediapipe as mp
import math
import collections
from HandTrackingModule import HandDetector
import random

# ==========================================
# 0. PARTICLE SYSTEM
# ==========================================
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.alpha = 255
        self.size = random.randint(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= 8 # Fade speed
        self.size -= 0.1

    def draw(self, img):
        if self.alpha > 0 and self.size > 0:
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), int(self.size), self.color, -1)
            cv2.addWeighted(overlay, self.alpha/255.0, img, 1 - self.alpha/255.0, 0, img)

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, color):
        for _ in range(2): # Emit 2 particles per frame
            self.particles.append(Particle(x, y, color))

    def update_and_draw(self, img):
        for p in self.particles[:]:
            p.update()
            if p.alpha <= 0 or p.size <= 0:
                self.particles.remove(p)
            else:
                p.draw(img)
# ==========================================
# 1. ROBUST SHAPE VALIDATOR (IoU Based)
# ==========================================
class ShapeValidator:
    def __init__(self):
        self.ref_size = (64, 64) # Smaller size for fuzziness logic

    def preprocess(self, img_bin):
        """Centers and resizes content to ref_size"""
        coords = cv2.findNonZero(img_bin)
        if coords is None: return None
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop
        roi = img_bin[y:y+h, x:x+w]
        
        # Resize to fit in ref_size with padding (preserve Aspect Ratio)
        target_h, target_w = self.ref_size
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        if new_w <= 0 or new_h <= 0: return None
        
        resized = cv2.resize(roi, (new_w, new_h))
        
        # Place in center of canvas
        canvas = np.zeros(self.ref_size, dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        
        canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
        return canvas

    def generate_reference(self, char):
        img = np.zeros((200, 200), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw large to get good resolution then shrink
        cv2.putText(img, char, (50, 150), font, 5, 255, 10)
        return self.preprocess(img)

    def validate(self, drawing_img, target_char, stroke_count):
        # 1. Preprocess Drawing
        gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        processed_draw = self.preprocess(bin_img)
        if processed_draw is None: return 0.0, "Empty"

        # 2. Generate Reference
        processed_ref = self.generate_reference(target_char)
        if processed_ref is None: return 0.0, "Error"
        
        # 3. IoU Calculation (Pixel Overlap)
        # Dilate drawing slightly to be forgiving
        kernel = np.ones((3,3), np.uint8)
        processed_draw = cv2.dilate(processed_draw, kernel, iterations=1)
        
        intersection = np.logical_and(processed_draw, processed_ref)
        union = np.logical_or(processed_draw, processed_ref)
        
        iou_score = np.sum(intersection) / np.sum(union)
        
        # Boost score slightly since humans aren't printers
        final_score = min(1.0, iou_score * 1.5) 
        
        return final_score, f"IoU: {iou_score:.2f}"

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
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.particles = ParticleSystem()
        
        self.wheel = AlphabetWheel(pos=(200, 200), radius=120)
        self.validator = ShapeValidator()
        
        # State
        self.sm_x, self.sm_y = 0, 0 # Smooth coords
        self.alpha = 0.3 # Smoothing factor (Lower = Smoother)
        
        self.hue = 0 # For rainbow color
        
        self.drawing = False
        self.strokes = [] # List of finished stroke paths [ (color, points), ... ]
        self.current_stroke = [] # Active stroke
        self.current_color = (255, 255, 255)

        
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
            # 1. Processing (Hand Tracking)
            img = self.detector.findHands(img)
            lmList = self.detector.findPosition(img, draw=False)
            
            x1, y1 = 0, 0
            fingers = []
            hand_detected = False
            
            if len(lmList) != 0:
                hand_detected = True
                self.last_hand_time = time.time() # Update valid time
                
                # Index Tip
                x1, y1 = lmList[8][1], lmList[8][2]
                # Fingers Up Check
                fingers = self.detector.fingersUp()

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
                    
                    # Update Color (Rainbow Cycle)
                    self.hue = (self.hue + 2) % 180
                    # Convert HSV to BGR for OpenCV
                    hw_color = np.uint8([[[self.hue, 255, 255]]])
                    bgr_color = cv2.cvtColor(hw_color, cv2.COLOR_HSV2BGR)[0][0]
                    self.current_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

                    if draw_point:
                        self.current_stroke.append(draw_point)
                        # Emit Particles
                        self.particles.emit(draw_point[0], draw_point[1], self.current_color)
                
                # STOP / VALIDATE (Index Down)
                # Ensure we only stop if we were actually drawing and we explicitly lower fingers
                elif self.drawing and not fingers[1]: 
                    self.drawing = False
                    if self.current_stroke:
                        # Only finalize if it has significant length to avoid dots
                        if len(self.current_stroke) > 2:
                            self.strokes.append((self.current_color, list(self.current_stroke)))
                            # Validate
                            # Create a temp canvas for OCR
                            temp_canvas = np.zeros((self.height, self.width, 3), np.uint8)
                            for col, s in self.strokes:
                                if len(s) > 1: cv2.polylines(temp_canvas, [np.array(s)], False, (255,255,255), 15)
                            
                            score, details = self.validator.validate(temp_canvas, target_char, len(self.strokes))
                            
                            match_res = "MATCH!" if score > 0.8 else "..."
                            color = (0, 255, 0) if score > 0.8 else (0, 0, 255)
                            self.validation_result = (target_char, score, match_res, color)
                            print(f"Target: {target_char} | Score: {score:.2f} | {details}")
                    self.current_stroke = [] # Clear active
                    
                # CLEAR (Open Palm)
                if all(fingers[1:]):
                     if self.strokes:
                         # Explosion effect
                         for _ in range(50):
                             self.particles.emit(self.width//2, self.height//2, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
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
                         self.strokes.append((self.current_color, list(self.current_stroke)))
                    self.current_stroke = []


            # 3. Rendering Pipeline
            
            # Dark Overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (self.width, self.height), (0,0,0), cv2.FILLED)
            cv2.addWeighted(img, 0.7, overlay, 0.3, 0, img)
            
            # Wheel
            self.wheel.draw(img)
            
            # Landmarks
            # Landmarks (Handled by findHands)

            # Strokes (Polylines for smoothness)
            # Draw Finished Strokes
            for col, s in self.strokes:
                if len(s) > 1:
                    # Color core
                    cv2.polylines(img, [np.array(s)], False, col, 15, cv2.LINE_AA)
                    # White Glow outline
                    cv2.polylines(img, [np.array(s)], False, (255, 255, 255), 4, cv2.LINE_AA)
            
            # Draw Current Stroke
            if len(self.current_stroke) > 1:
                cv2.polylines(img, [np.array(self.current_stroke)], False, self.current_color, 15, cv2.LINE_AA)
                cv2.polylines(img, [np.array(self.current_stroke)], False, (255, 255, 255), 4, cv2.LINE_AA) # White Glow

            # Draw Particles
            self.particles.update_and_draw(img)

            
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



if __name__ == "__main__":
    app = App()
    app.run()
