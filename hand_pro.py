import cv2
import numpy as np
import math
import time

# ================================
# CONFIGURATION
# ================================

# Put YOUR tuned HSV values here:
LOWER_COLOR = np.array([0, 45, 60])      # e.g. from hsv_tuning.py
UPPER_COLOR = np.array([17, 255, 255])   # e.g. from hsv_tuning.py

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WARNING_DISTANCE = 120.0   # pixels
DANGER_DISTANCE = 40.0     # pixels

MIN_HAND_AREA = 400        # ignore tiny blobs
MAX_HAND_AREA = 40000      # ignore huge blobs (torso/wall)

SMOOTHING_ALPHA = 0.6      # 0..1 (0.6 = fairly smooth but responsive)
HAND_TIMEOUT_FRAMES = 12   # keep last hand pos for these many missed frames
HAND_ROI_TOP_FRACTION = 0.35  # ignore top 35% of frame (face region)


# ================================
# UTILITY: distance from point to rectangle
# ================================
def point_to_rect_distance(px, py, x1, y1, x2, y2):
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


# ================================
# HAND TRACKER (color + ROI + smoothing)
# ================================
class HandTracker:
    def __init__(self,
                 lower_hsv,
                 upper_hsv,
                 min_area=MIN_HAND_AREA,
                 max_area=MAX_HAND_AREA,
                 alpha=SMOOTHING_ALPHA,
                 timeout_frames=HAND_TIMEOUT_FRAMES,
                 roi_top_frac=HAND_ROI_TOP_FRACTION):

        self.lower = lower_hsv
        self.upper = upper_hsv
        self.min_area = min_area
        self.max_area = max_area
        self.alpha = alpha
        self.timeout_frames = timeout_frames
        self.roi_top_frac = roi_top_frac

        self.last_x = None
        self.last_y = None
        self.missed = 0

        self.kernel = np.ones((5, 5), np.uint8)

    def update(self, frame):
        """
        Returns (hand_present, (x, y) or None).
        Uses only color + ROI, with smoothing and persistence.
        """
        h, w, _ = frame.shape
        roi_top = int(h * self.roi_top_frac)

        # Only search for hand in lower part of frame to avoid face
        roi = frame[roi_top:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Clean mask
        mask = cv2.erode(mask, self.kernel, 1)
        mask = cv2.dilate(mask, self.kernel, 2)

        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        hand_present = False

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)

            if self.min_area < area < self.max_area:
                M = cv2.moments(biggest)
                if M["m00"] != 0:
                    raw_x = int(M["m10"] / M["m00"])
                    raw_y = int(M["m01"] / M["m00"])

                    # Convert ROI coords → full-frame coords
                    raw_y += roi_top

                    # Smooth position
                    if self.last_x is None:
                        self.last_x, self.last_y = raw_x, raw_y
                    else:
                        self.last_x = int(self.last_x * self.alpha +
                                          raw_x * (1 - self.alpha))
                        self.last_y = int(self.last_y * self.alpha +
                                          raw_y * (1 - self.alpha))

                    hand_present = True
                    self.missed = 0

        if not hand_present:
            # keep showing last point for a while so dot doesn't disappear instantly
            self.missed += 1
            if self.last_x is not None and self.missed <= self.timeout_frames:
                hand_present = True
            else:
                self.last_x = None
                self.last_y = None

        if hand_present and self.last_x is not None:
            return True, (self.last_x, self.last_y)
        else:
            return False, None


# ================================
# MAIN APPLICATION
# ================================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tracker = HandTracker(LOWER_COLOR, UPPER_COLOR)

    # Virtual danger-zone rectangle in center
    rw = 200
    rh = 150
    rx1 = FRAME_WIDTH // 2 - rw // 2
    ry1 = FRAME_HEIGHT // 2 - rh // 2
    rx2 = rx1 + rw
    ry2 = ry1 + rh

    prev_time = time.time()
    fps = 0.0

    print("Running danger zone demo. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        frame = cv2.flip(frame, 1)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt

        # ---- HAND TRACKING ----
        hand_present, hand_pos = tracker.update(frame)

        # ---- DISTANCE & STATE ----
        if hand_present and hand_pos is not None:
            hx, hy = hand_pos
            distance = point_to_rect_distance(hx, hy, rx1, ry1, rx2, ry2)
        else:
            distance = 9999.0

        if distance <= DANGER_DISTANCE:
            state = "DANGER"
            color = (0, 0, 255)
        elif distance <= WARNING_DISTANCE:
            state = "WARNING"
            color = (0, 165, 255)
        else:
            state = "SAFE"
            color = (0, 255, 0)

        # ---- DRAW OVERLAYS ----

        # Virtual boundary
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 3)

        # Hand dot (now smoother & persistent)
        if hand_present and hand_pos is not None:
            cv2.circle(frame, hand_pos, 6, (255, 255, 255), -1)

        # State text
        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Distance or info
        if hand_present and hand_pos is not None:
            cv2.putText(frame, f"Dist: {distance:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200), 2)

        # FPS (small)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (FRAME_WIDTH - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # Big warning/danger overlays
        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER",
                        (60, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (0, 0, 255), 4)
        elif state == "WARNING":
            cv2.putText(frame, "Approaching boundary",
                        (60, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 165, 255), 3)

        cv2.imshow("Danger Zone Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
