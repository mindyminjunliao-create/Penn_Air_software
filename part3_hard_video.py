import os
import pickle
import cv2
import numpy as np
from src.detector import detect_objects_rgb_and_noise, sample_background_bgr  # Import both functions

# Directory where per-frame process images will be saved
SAVE_DIR = "outputs_rgb_part3"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the target video file
cap = cv2.VideoCapture("assets/PennAir 2024 App Dynamic Hard.mp4")


def extract_noise_pattern(gray_img, ksize=5):
    """Mirrors the noise-extraction logic in detector.py so the background
    noise reference is computed the exact same way it will later be compared."""
    blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    high_freq_noise = cv2.absdiff(gray_img, blurred)
    noise_std = cv2.GaussianBlur(high_freq_noise.astype(np.float32) ** 2, (ksize, ksize), 0)
    return np.sqrt(np.maximum(noise_std, 0))


# Grab the first frame to estimate the background's color AND noise pattern
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame from the video file.")

# --- CHANGED: bg_bgr is now sampled automatically instead of hardcoded ---
# Assumes the object(s) don't touch the frame's four corners on frame 0.
# If that assumption doesn't hold for a given video, corners_only=False
# switches to sampling thin border strips on all 4 edges instead, which is
# more robust as long as the object doesn't touch the frame edges.
bg_bgr = sample_background_bgr(first_frame, patch_size=20, corners_only=True)
print(f"[Part 3] Auto-sampled background BGR: {bg_bgr}")

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
bg_noise_std_mean = extract_noise_pattern(first_gray)

# Rewind so the first frame still gets processed in the main loop below
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Run background-agnostic detection on the current frame
    processed_frame, centers, shape_contours, clustered_mask, color_mask, noise_mask = detect_objects_rgb_and_noise(
        frame, bg_bgr=bg_bgr, bg_noise_std_mean=bg_noise_std_mean
    )

    # Only save the process images for the very first frame (frame 0)
    if frame_idx == 0:
        prefix = "frame0"
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_0_original.png"), frame)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_1_color_mask.png"), color_mask)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_2_noise_mask.png"), noise_mask)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_3_clustered_mask.png"), clustered_mask)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_4_result.png"), processed_frame)

        # Persist the detected centers + their contours so part4_3d.py can later
        # take a manually-typed (x, y) from this frame's saved image, look up
        # which detected shape it belongs to, and measure its pixel radius.
        frame0_shapes = {"centers": centers, "contours": shape_contours}
        with open(os.path.join(SAVE_DIR, f"{prefix}_shapes.pkl"), "wb") as f:
            pickle.dump(frame0_shapes, f)
        print(f"[Part 3] Saved {len(centers)} frame-0 shape(s) to "
              f"{os.path.join(SAVE_DIR, prefix + '_shapes.pkl')}")
        for i, c in enumerate(centers):
            print(f"    shape {i}: center={c}")

    frame_idx += 1

    # Display the processed frame with detection overlays
    cv2.imshow("Part 3 - Background Agnostic Video Detection", processed_frame)

    # Press 'q' to exit the video loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up video resources and close GUI windows
cap.release()
cv2.destroyAllWindows()
