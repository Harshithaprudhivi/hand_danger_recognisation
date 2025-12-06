import cv2

print("Starting camera_web.py...")

def main():
    print("Inside main()")

    # Try camera index 0 first
    print("Trying to open camera with index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows

    if not cap.isOpened():
        print("Could not open camera index 0. Trying index 1...")
        cap.release()
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open any camera (index 0 or 1).")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera Web Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Detected 'q' key. Exiting loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and closed windows. Exiting program.")

if __name__ == "__main__":
    print("__name__ is __main__, calling main()")
    main()
