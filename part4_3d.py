import cv2
import numpy as np

# Define camera intrinsic matrix K provided in the challenge
# K = [[fx,  0, cx],
#      [ 0, fy, cy],
#      [ 0,  0,  1]]
K = np.array([
    [2564.3186869, 0, 0],
    [0, 2569.70273111, 0],
    [0, 0, 1]
], dtype=np.float64)

# Extract focal lengths from the intrinsic matrix
fx = K[0, 0]
fy = K[1, 1]

# Known physical dimensions (in inches)
REAL_RADIUS_INCHES = 10.0

# Example inputs: Detected circle center (u_pixel, v_pixel) and its radius in pixels
# Replace these values with the output from your 2D shape detector
u_pixel = 1280.0  # Center X coordinate in pixel space
v_pixel = 720.0   # Center Y coordinate in pixel space
r_pixel = 50.0    # Detected radius of the circle in pixels

# Step 1: Calculate depth (Z coordinate) using pinhole camera model
# Formula: Depth (Z) = (focal_length * real_world_radius) / pixel_radius
# Using the average focal length (fx + fy) / 2 for balanced scaling
f_avg = (fx + fy) / 2.0
depth_z = (f_avg * REAL_RADIUS_INCHES) / r_pixel

# Step 2: Compute 3D coordinates (X, Y) relative to the camera
# Formula: X = (u * Z) / fx,  Y = (v * Z) / fy  (since cx=0, cy=0 in this matrix)
x_3d = (u_pixel * depth_z) / fx
y_3d = (v_pixel * depth_z) / fy

# Step 3: Output the resulting 3D coordinates (X, Y, Z) in inches
print(f"Detected Center 3D Coordinates (w.r.t. camera):")
print(f"X: {x_3d:.2f} in")
print(f"Y: {y_3d:.2f} in")
print(f"Z (Depth): {depth_z:.2f} in")
