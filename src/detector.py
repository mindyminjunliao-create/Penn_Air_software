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
