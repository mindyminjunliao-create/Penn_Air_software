import cv2
import numpy as np

def detect_shapes_by_color(frame, bg_color_hsv_lower, bg_color_hsv_upper):
    """
    Detects objects by subtracting a known background color in HSV space.
    
    bg_color_hsv_lower: Lower bound for background color (e.g., np.array([35, 40, 40]) for green)
    bg_color_hsv_upper: Upper bound for background color (e.g., np.array([85, 255, 255]))
    """
    output_frame = frame.copy()

    # 1. Convert BGR frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Create a mask where background pixels are white (255) and others are black (0)
    bg_mask = cv2.inRange(hsv, bg_color_hsv_lower, bg_color_hsv_upper)

    # 3. Invert mask: Non-background objects become white (255)
    object_mask = cv2.bitwise_not(bg_mask)

    # 4. Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    # Opening removes small noise spots
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    # Closing fills small holes inside detected objects
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours of non-background shapes
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        # Filter out small contours/noise by area
        if cv2.contourArea(cnt) < 200:
            continue

        # Draw contour outlines on output frame
        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)

        # Calculate centroid using spatial moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            # Draw center point and text coordinates
            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(
                output_frame, f"({cX},{cY})", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    return output_frame, centers, object_mask
