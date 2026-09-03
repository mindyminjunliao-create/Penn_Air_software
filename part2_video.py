import cv2
import numpy as np
from src.detector import detect_shapes_with_grayscale_mask

cap = cv2.VideoCapture("assets/PennAir 2024 App Dynamic.mp4")

lower_hsv = np.array([0, 0, 0])
# Upper HSV boundary: [Hue=105, Saturation=243, Value=155]
upper_hsv = np.array([105, 243, 155])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Process frame frame-by-frame
    processed_frame, centers, mask = detect_shapes_with_grayscale_mask(frame,lower_hsv, upper_hsv)

    cv2.imshow("Part 2 - Video Stream Shape Detection", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
