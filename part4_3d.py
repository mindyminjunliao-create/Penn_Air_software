import cv2
import numpy as np
from src.detector import detect_shapes_in_frame

# Camera Intrinsic Matrix K from prompt
K = np.array([
    [2564.3186869, 0.0, 0.0],
    [0.0, 2569.70273111, 0.0],
    [0.0, 0.0, 1.0]
])

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# Physical known size (inches)
R_REAL = 10.0  # circle radius in inches

img = cv2.imread("assets/PennAir 2024 App Static.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = img.copy()

for cnt in contours:
    if cv2.contourArea(cnt) < 150:
        continue

    # Fit minimum enclosing circle to estimate pixel radius
    (u, v), r_pixel = cv2.minEnclosingCircle(cnt)
    
    if r_pixel > 0:
        # 1. Calculate Depth (Z) using average focal length and known radius
        f_avg = (fx + fy) / 2.0
        Z = (f_avg * R_REAL) / r_pixel  # Depth in inches

        # 2. Map 2D pixel coordinates (u, v) to 3D camera coordinates (X, Y, Z)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # 3. Draw onto image
        center_pt = (int(u), int(v))
        cv2.circle(output_img, center_pt, 4, (0, 0, 255), -1)
        
        label = f"X:{X:.1f}in Y:{Y:.1f}in Z:{Z:.1f}in"
        cv2.putText(
            output_img, label, (int(u) - 40, int(v) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        
        print(f"Detected Shape at Pixel ({u:.1f}, {v:.1f}) -> 3D Camera Frame: X={X:.2f}\", Y={Y:.2f}\", Z={Z:.2f}\"")

cv2.imwrite("outputs/part4_3d_result.png", output_img)
