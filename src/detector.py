import cv2
import numpy as np

def detect_shapes_in_frame(frame, background_agnostic=False):
    """
    processes a single frame to trace shape outlines and returns:
        processed_frame: Image with drawn outlines and center markers
        centers: List of tuples [(x,y), ...] for detected shapes
    """

    output_frame = frame.copy()

    #convert to grayscale, then apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #edge detection
    if background_agnostic:
        edges = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
         )
    else:
        edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 150:
            continue

        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            cv2.circle(output_frame, (cX, cY), 4, (0,0,255), -1)
            cv2.putText(
                output_frame, f"({cX}, {cY})", (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
    
    return output_frame, centers

