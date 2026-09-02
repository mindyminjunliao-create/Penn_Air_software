import cv2
from src.detector import detect_shapes_in_frame

img = cv2.imread("assets/PennAir 2024 App Static.png")
if img is None:
    raise FileNotFoundError("Image not found in assets/ directory.")

processed_img, centers = detect_shapes_in_frame(img, background_agnostic=False)

cv2.imwrite("outputs/part1_result.png", processed_img)
print(f"Part 1 Complete. Detected {len(centers)} shapes.")
