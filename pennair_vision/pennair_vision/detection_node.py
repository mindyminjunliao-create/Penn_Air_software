import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

# Import core detection function from detector.py
from pennair_vision.detector import detect_objects_rgb_and_noise

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.coord_pub = self.create_publisher(PointStamped, '/target_3d_pose', 10)
        self.bridge = CvBridge()

        # Part 4 camera intrinsic and physical parameters definition
        self.K = np.array([
            [2564.3186869, 0, 0],
            [0, 2569.70273111, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.f_avg = (self.fx + self.fy) / 2.0
        self.REAL_RADIUS_INCHES = 10.0
        self.bg_bgr = np.array([37.7, 37.7, 37.7], dtype=np.float32)

    def image_callback(self, msg):
        # 1. Convert ROS Image to OpenCV image, using 'passthrough' to avoid encoding errors
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # 2. Run Part 3 background-agnostic RGB & Noise detection algorithm
        bg_noise_std_mean = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        processed_frame, centers, shape_contours, _, _, _ = detect_objects_rgb_and_noise(
            frame, bg_bgr=self.bg_bgr, bg_noise_std_mean=bg_noise_std_mean
        )

        # 3. Run Part 4 3D PnP / depth coordinate conversion
        if centers and shape_contours:
            # Select the largest contour as the reference circle/object
            ref_contour = max(shape_contours, key=cv2.contourArea)
            (_, _), ref_r_pixel = cv2.minEnclosingCircle(ref_contour)

            if ref_r_pixel > 0:
                # Pinhole camera model for depth (Z) calculation
                depth_z = (self.f_avg * self.REAL_RADIUS_INCHES) / ref_r_pixel
                
                # Get 2D center pixel coordinates (u, v)
                cX, cY = centers[0]
                
                # Map to 3D camera coordinate system (X, Y, Z)
                x_3d = (float(cX) * depth_z) / self.fx
                y_3d = (float(cY) * depth_z) / self.fy

                # 4. Construct and publish PointStamped 3D pose message
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "camera_frame"
                point_msg.point.x = float(x_3d)
                point_msg.point.y = float(y_3d)
                point_msg.point.z = float(depth_z)
                self.coord_pub.publish(point_msg)
                
                self.get_logger().info(f'Published 3D Pose: X={x_3d:.2f}in, Y={y_3d:.2f}in, Z={depth_z:.2f}in')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
