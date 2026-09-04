import os
import cv2
import numpy as np

def detect_shapes_with_grayscale_mask(frame, lower_hsv, upper_hsv, save_dir="outputs_hsv_process"):
    os.makedirs(save_dir, exist_ok=True)

    # 0. Save original input image
    cv2.imwrite(os.path.join(save_dir, "0_original.png"), frame)

    # Create a copy of the original frame for visualization overlay
    output_frame = frame.copy()

    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Generate raw HSV mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cv2.imwrite(os.path.join(save_dir, "1_hsv_mask.png"), mask)

    # Invert mask: convert black blocks (0) to white (255) for contour extraction
    black_blocks_mask = cv2.bitwise_not(mask)
    cv2.imwrite(os.path.join(save_dir, "1_black_blocks_mask.png"), black_blocks_mask)

    # 3. Find contours of target blocks
    contours, _ = cv2.findContours(black_blocks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create single-channel image for drawing contours
    mask_contours_img = np.zeros_like(mask)

    centers = []

    # Area filtering thresholds
    min_area = 500   # Ignore small noise artifacts
    max_area = (frame.shape[0] * frame.shape[1]) * 0.9  # Ignore whole-frame background contours

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter out contours based on area limits
        if area < min_area or area > max_area:
            continue

        # 4a. Draw contour on single-channel mask image
        cv2.drawContours(mask_contours_img, [cnt], -1, 255, 2)

        # 4b. Draw green contour on the original frame overlay
        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)

        # Calculate geometric center (centroid)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            # Draw center point and coordinates text
            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(
                output_frame, f"({cX},{cY})", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    cv2.imwrite(os.path.join(save_dir, "2_mask_contours.png"), mask_contours_img)
    cv2.imwrite(os.path.join(save_dir, "3_final_overlay.png"), output_frame)

    return output_frame, centers, mask


def detect_objects_rgb_and_noise(
    frame, bg_bgr, bg_noise_std_mean, 
    color_thresh=70,          # Higher threshold prevents full-white foreground masks
    noise_diff_thresh=12.0
):
    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. BGR Color Distance Mask
    color_diff = np.linalg.norm(frame.astype(np.float32) - bg_bgr, axis=2)
    fg_color_mask = (color_diff > color_thresh).astype(np.uint8) * 255

    # Apply morphological CLOSE to eliminate internal black gaps/gradient breaks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_color_mask_closed = cv2.morphologyEx(fg_color_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2. High-Frequency Noise Pattern Mask
    def extract_noise_pattern(gray_img, ksize=5):
        blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
        high_freq_noise = cv2.absdiff(gray_img, blurred)
        noise_std = cv2.GaussianBlur(high_freq_noise.astype(np.float32)**2, (ksize, ksize), 0)
        return np.sqrt(np.maximum(noise_std, 0))

    current_noise_std = extract_noise_pattern(gray)
    noise_diff = np.abs(current_noise_std - bg_noise_std_mean)
    fg_noise_mask = (noise_diff > noise_diff_thresh).astype(np.uint8) * 255

    # Dilate noise mask to connect sparse/isolated white noise points
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_noise_mask_dilated = cv2.dilate(fg_noise_mask, kernel_dilate, iterations=2)

    # 3. Combine Color and Noise Masks
    combined_fg_mask = cv2.bitwise_or(fg_color_mask_closed, fg_noise_mask_dilated)

    # Perform MORPH_OPEN to clear tiny artifacts, followed by MORPH_CLOSE for clustering
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_mask = cv2.morphologyEx(combined_fg_mask, cv2.MORPH_OPEN, kernel_clean)
    
    kernel_cluster = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clustered_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_cluster)

    # 4. Find External Contours
    contours, _ = cv2.findContours(clustered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    h, w = frame.shape[:2]
    min_area = 400
    max_area = (h * w) * 0.7  # Exclude full-frame contours to prevent over-segmentation

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        hull = cv2.convexHull(cnt)
        cv2.drawContours(output_frame, [hull], -1, (0, 255, 0), 2)

        M = cv2.moments(hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(
                output_frame, f"({cX},{cY})", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    return output_frame, centers, clustered_mask, fg_color_mask_closed, fg_noise_mask_dilated
