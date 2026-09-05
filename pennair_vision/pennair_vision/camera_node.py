import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        # Create a publisher on topic /camera/image_raw
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        # Create a timer running at ~30 FPS
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.bridge = CvBridge()
        
        # Path to the actual video file
        self.video_path = "/ros2_ws/assets/PennAir 2024 App Dynamic Hard.mp4"
        self.cap = cv2.VideoCapture(self.video_path)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            # Loop back to the first frame when the video reaches the end
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            # Convert OpenCV image to ROS Image message using 'passthrough' encoding
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='passthrough')
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing image frame...')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
