import os
import numpy as np
import cv2
from src.detector import detect_shapes_by_color

frame = cv2.imread("assets/PennAir 2024 App Static.png")
output_frame, centers, object_mask = detect_shapes_by_color(frame)

os.makedirs("outputs_hsv", exist_ok=True)
cv2.imwrite("outputs_hsv/part1_result_mask.png", output_frame)
