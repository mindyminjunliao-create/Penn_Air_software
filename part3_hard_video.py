import cv2
import numpy as np
from src.detector import detect_objects_rgb_and_noise  # Import the existing detection function

# Load the target video file
cap = cv2.VideoCapture("assets/PennAir 2024 App Dynamic Hard.mp4")

# Define the baseline background color in BGR format
bg_bgr = np.array([37.7, 37.7, 37.7], dtype=np.float32)


def extract_noise_pattern(gray_img, ksize=5):
    """Mirrors the noise-extraction logic in detector.py so the background
    noise reference is computed the exact same way it will later be compared."""
    blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    high_freq_noise = cv2.absdiff(gray_img, blurred)
    noise_std = cv2.GaussianBlur(high_freq_noise.astype(np.float32) ** 2, (ksize, ksize), 0)
    return np.sqrt(np.maximum(noise_std, 0))


# Grab the first frame to estimate the background's noise pattern
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read the first frame from the video file.")

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
bg_noise_std_mean = extract_noise_pattern(first_gray)

# Rewind so the first frame still gets processed in the main loop below
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Run background-agnostic detection on the current frame
    processed_frame, centers, clustered_mask, color_mask, noise_mask = detect_objects_rgb_and_noise(
        frame, bg_bgr=bg_bgr, bg_noise_std_mean=bg_noise_std_mean
    )

    # Display the processed frame with detection overlays
    cv2.imshow("Part 3 - Background Agnostic Video Detection", processed_frame)

    # Press 'q' to exit the video loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up video resources and close GUI windows
cap.release()
cv2.destroyAllWindows()
