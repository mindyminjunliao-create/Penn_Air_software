import os
import numpy as np
import cv2
from src.detector import detect_shapes_with_grayscale_mask

# Lower HSV boundary: [Hue=0, Saturation=0, Value=0]
lower_hsv = np.array([0, 0, 0])

# Upper HSV boundary: [Hue=105, Saturation=243, Value=155]
upper_hsv = np.array([105, 243, 155])

frame = cv2.imread("assets/PennAir 2024 App Static.png")
output_frame, centers, mask = detect_shapes_with_grayscale_mask(frame, lower_hsv, upper_hsv)

os.makedirs("outputs_hsv", exist_ok=True)
cv2.imwrite("outputs_hsv/part1_result_mask_2.png", output_frame)
