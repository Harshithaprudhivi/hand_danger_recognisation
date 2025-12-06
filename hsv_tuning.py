import cv2
import numpy as np

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera")
        return

    cv2.namedWindow("HSV Tuner")
    cv2.resizeWindow("HSV Tuner", 400, 300)

    # Initial guesses – OK for skin in normal lighting
    cv2.createTrackbar("H Min", "HSV Tuner", 0,   179, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", 25,  179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", 30,  255, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", 60,  255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

    print("Move the sliders until your HAND appears white in the Mask window.")
    print("Press 'p' to print current HSV range, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Small cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, 1)
        mask = cv2.dilate(mask, kernel, 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            print("LOWER_COLOR =", lower, "UPPER_COLOR =", upper)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
