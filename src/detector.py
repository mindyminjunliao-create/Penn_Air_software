import os
import cv2
import numpy as np

def detect_shapes_with_grayscale_mask(frame, lower_hsv, upper_hsv, save_dir="outputs_hsv_process"):
    os.makedirs(save_dir, exist_ok=True)

    # 0. 保存原始输入图片
    cv2.imwrite(os.path.join(save_dir, "0_original.png"), frame)

    # 创建原图副本用于叠加绘制
    output_frame = frame.copy()

    # 1. 转换至 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 生成原始 HSV Mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cv2.imwrite(os.path.join(save_dir, "1_hsv_mask.png"), mask)

    # 关键修改：对 mask 取反，将黑色色块（0）转为白色（255），以便 findContours 提取
    black_blocks_mask = cv2.bitwise_not(mask)
    cv2.imwrite(os.path.join(save_dir, "1_black_blocks_mask.png"), black_blocks_mask)

    # 3. 查找黑色色块的轮廓
    contours, _ = cv2.findContours(black_blocks_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建专门保存轮廓的单通道图
    mask_contours_img = np.zeros_like(mask)

    centers = []
    
    # 面积过滤阈值设置
    min_area = 500   # 忽略面积过小的黑色噪点/杂色块
    max_area = (frame.shape[0] * frame.shape[1]) * 0.9  # 忽略整个背景大框

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 过滤掉面积过小或过大的黑色色块
        if area < min_area or area > max_area:
            continue

        # 4a. 绘制轮廓到单通道图
        cv2.drawContours(mask_contours_img, [cnt], -1, 255, 2)

        # 4b. 绘制绿色轮廓到原图副本
        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)

        # 计算几何中心（质心）
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

            # 在轮廓图和原图副本标出中心点
            cv2.circle(mask_contours_img, (cX, cY), 5, 255, -1)
            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(
                output_frame, f"({cX},{cY})", (cX - 20, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    # 5. 保存仅包含 Mask 轮廓的图片和最终叠加图
    cv2.imwrite(os.path.join(save_dir, "2_mask_contours.png"), mask_contours_img)
    cv2.imwrite(os.path.join(save_dir, "3_final_overlay.png"), output_frame)

    return output_frame, centers, mask

def estimate_background_colors(frame, grid_size=20):
    """
    Samples the first frame using grid cells to find the dominant background color.
    Returns both BGR and HSV representations for analysis/logging.
    """
    h, w, _ = frame.shape
    grid_colors_bgr = []

    for y in range(0, h - grid_size + 1, grid_size):
        for x in range(0, w - grid_size + 1, grid_size):
            cell = frame[y:y+grid_size, x:x+grid_size]
            mean_val = cell.mean(axis=(0, 1))  # BGR mean
            grid_colors_bgr.append(mean_val)

    grid_colors_bgr = np.array(grid_colors_bgr, dtype=np.float32)

    # Quantize colors (bin size = 16) to group similar background blocks
    quantized_colors = (grid_colors_bgr // 16) * 16
    unique_colors, counts = np.unique(quantized_colors, axis=0, return_counts=True)
    dominant_quantized = unique_colors[np.argmax(counts)]

    # Compute exact average BGR of the dominant background blocks
    mask = np.all(quantized_colors == dominant_quantized, axis=1)
    bg_bgr = grid_colors_bgr[mask].mean(axis=0)

    # Convert dominant BGR to HSV for reference
    bg_pixel_bgr = np.uint8([[bg_bgr]])
    bg_hsv = cv2.cvtColor(bg_pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]

    return bg_bgr, bg_hsv

def extract_noise_pattern(gray_img, ksize=5):
    """
    Computes local noise/texture intensity map using high-frequency residuals.
    """
    blurred = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
    high_freq_noise = cv2.absdiff(gray_img, blurred)
    noise_std = cv2.GaussianBlur(high_freq_noise.astype(np.float32)**2, (ksize, ksize), 0)
    noise_std = np.sqrt(np.maximum(noise_std, 0))
    return noise_std

def detect_objects_rgb_and_noise(
    frame, bg_bgr, bg_noise_std_mean, 
    color_thresh=25, noise_diff_thresh=8.0
):
    """
    RGB/BGR color distance based object detection with noise pattern matching.
    """
    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. BGR Euclidean distance from estimated background
    color_diff = np.linalg.norm(frame.astype(np.float32) - bg_bgr, axis=2)
    fg_color_mask = (color_diff > color_thresh).astype(np.uint8) * 255

    # 2. Noise pattern discrepancy
    current_noise_std = extract_noise_pattern(gray)
    noise_diff = np.abs(current_noise_std - bg_noise_std_mean)
    fg_noise_mask = (noise_diff > noise_diff_thresh).astype(np.uint8) * 255

    # 3. Combine masks
    combined_fg_mask = cv2.bitwise_or(fg_color_mask, fg_noise_mask)

    # 4. Morphological clustering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    clustered_mask = cv2.morphologyEx(combined_fg_mask, cv2.MORPH_CLOSE, kernel)
    clustered_mask = cv2.dilate(clustered_mask, kernel, iterations=2)

    # 5. Extract outer boundaries
    contours, _ = cv2.findContours(clustered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    min_area = 300
    max_area = (frame.shape[0] * frame.shape[1]) * 0.85

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

    return output_frame, centers, clustered_mask, fg_color_mask, fg_noise_mask
