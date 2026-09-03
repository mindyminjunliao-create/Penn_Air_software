import cv2
import numpy as np

def detect_shapes_by_color(frame):
    output_frame = frame.copy()

    # 1. Convert BGR frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper HSV values obtained from the web tool
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([105, 243, 155])

    # 2. Create the mask using inRange
    object_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 3. Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Find contours of detected shapes
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue

        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(
                output_frame, f"({cX},{cY})", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    return output_frame, centers, object_mask
