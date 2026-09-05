import os
import pickle
import cv2
import numpy as np

# Directory where part3_hard_video.py stored frame-0's process images and
# the pickled {"centers": [...], "contours": [...]} data for that frame.
PART3_SAVE_DIR = "outputs_rgb_part3"
FRAME0_IMAGE = os.path.join(PART3_SAVE_DIR, "frame0_0_original.png")
FRAME0_SHAPES_PKL = os.path.join(PART3_SAVE_DIR, "frame0_shapes.pkl")

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
f_avg = (fx + fy) / 2.0

# Known physical radius of the *reference circle* only (in inches), per the
# challenge statement. Other shapes' real-world size is NOT known -- we do
# NOT assume they are 10in circles too.
REAL_RADIUS_INCHES = 10.0


def load_frame0_shapes():
    """Load the centers + contours that part3_hard_video.py detected on frame 0."""
    if not os.path.exists(FRAME0_SHAPES_PKL):
        raise FileNotFoundError(
            f"Could not find '{FRAME0_SHAPES_PKL}'. Run the updated "
            f"part3_hard_video.py first so it saves frame 0's detections."
        )
    with open(FRAME0_SHAPES_PKL, "rb") as f:
        data = pickle.load(f)
    return data["centers"], data["contours"]


def prompt_for_point():
    """Ask the user to type the (x, y) pixel coordinates of the KNOWN circle's
    center (the one whose real-world radius = 10in is given in the problem),
    as read off the saved frame0_0_original.png / frame0_4_result.png image."""
    raw = input(
        "Enter the (x, y) pixel coordinates of the KNOWN 10in-radius circle's "
        f"center from '{FRAME0_IMAGE}' (e.g. '640,360' or '640 360'): "
    )
    raw = raw.replace(",", " ").split()
    if len(raw) != 2:
        raise ValueError(f"Expected two numbers, got: {raw}")
    return float(raw[0]), float(raw[1])


def find_matching_shape(u_click, v_click, centers, contours):
    """Match the manually-entered point to one of part3's detected shapes.

    First choice: the contour whose polygon actually contains the clicked
    point (cv2.pointPolygonTest). Falls back to the nearest detected center
    if the click didn't land exactly inside any contour.
    Returns (index, center, contour).
    """
    point = (float(u_click), float(v_click))

    for idx, cnt in enumerate(contours):
        if cv2.pointPolygonTest(cnt, point, False) >= 0:
            return idx, centers[idx], cnt

    if not centers:
        raise ValueError("No shapes were detected in frame 0 by part3_hard_video.py.")

    dists = [np.hypot(cx - u_click, cy - v_click) for (cx, cy) in centers]
    idx = int(np.argmin(dists))
    print(
        f"[Note] Click didn't land inside any detected contour; using the "
        f"nearest detected shape's center {centers[idx]} "
        f"(distance = {dists[idx]:.1f} px) instead."
    )
    return idx, centers[idx], contours[idx]


def main():
    centers, contours = load_frame0_shapes()
    print(f"Loaded {len(centers)} shape(s) detected by part3 on frame 0: {centers}\n")

    # --- Step 1: identify the KNOWN circle (real radius = 10 in) by manual input ---
    u_click, v_click = prompt_for_point()
    ref_idx, (ref_cX, ref_cY), ref_contour = find_matching_shape(
        u_click, v_click, centers, contours
    )
    (_, _), ref_r_pixel = cv2.minEnclosingCircle(ref_contour)
    print(
        f"Reference circle = shape #{ref_idx}, center=({ref_cX}, {ref_cY}) px, "
        f"pixel radius={ref_r_pixel:.2f} px\n"
    )

    # --- Step 2: use the known circle to solve for the (shared) plane depth ---
    # Pinhole model: Z = (f_avg * real_radius) / r_pixel
    # "Assume a flat surface" -> every shape in this frame lies on the SAME
    # plane, so this single depth_z applies to ALL of them, not just the circle.
    depth_z = (f_avg * REAL_RADIUS_INCHES) / ref_r_pixel
    print(f"Solved plane depth from the known circle: Z = {depth_z:.2f} in\n")

    # --- Step 3: generalize -- reuse depth_z to get every shape's (X, Y, Z) ---
    results = []
    for idx, (cX, cY) in enumerate(centers):
        x_3d = (float(cX) * depth_z) / fx
        y_3d = (float(cY) * depth_z) / fy
        results.append({
            "index": idx, "u_pixel": cX, "v_pixel": cY,
            "X": x_3d, "Y": y_3d, "Z": depth_z,
            "is_reference_circle": idx == ref_idx,
        })

        tag = "  <-- known reference circle" if idx == ref_idx else ""
        print(f"Shape #{idx}{tag}:")
        print(f"  2D center (u, v) = ({cX}, {cY}) px")
        print(f"  3D Coordinates (w.r.t. camera):")
        print(f"    X: {x_3d:.2f} in")
        print(f"    Y: {y_3d:.2f} in")
        print(f"    Z (Depth): {depth_z:.2f} in")
        print()

    return results


if __name__ == "__main__":
    main()
