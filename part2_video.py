import cv2
from src.detector import detect_shapes_in_frame

cap = cv2.VideoCapture("assets/PennAir 2024 App Dynamic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Process frame frame-by-frame
    processed_frame, centers = detect_shapes_in_frame(frame, background_agnostic=False)

    cv2.imshow("Part 2 - Video Stream Shape Detection", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
