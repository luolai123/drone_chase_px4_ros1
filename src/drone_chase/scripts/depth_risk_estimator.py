#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

from drone_chase.msg import DepthRisk


class DepthRiskEstimator:
    def __init__(self):
        rospy.init_node("depth_risk_estimator")

        self.min_depth = rospy.get_param("~min_depth", 0.2)
        self.max_depth = rospy.get_param("~max_depth", 10.0)
        self.safe_depth = rospy.get_param("~safe_depth", 0.8)
        self.danger_depth = rospy.get_param("~danger_depth", 0.5)
        self.danger_area_ratio = rospy.get_param("~danger_area_ratio", 0.2)
        self.min_valid_ratio = rospy.get_param("~min_valid_ratio", 0.02)
        self.roi_y_min_ratio = rospy.get_param("~roi_y_min_ratio", 0.15)
        self.roi_y_max_ratio = rospy.get_param("~roi_y_max_ratio", 0.70)
        self.use_altitude_gate = rospy.get_param("~use_altitude_gate", True)
        self.min_active_height = rospy.get_param("~min_active_height", 0.6)
        self.debug = rospy.get_param("~debug", True)

        self.bridge = CvBridge()
        self.latest_drone_z = None
        self.risk_pub = rospy.Publisher("/obstacle/risk", DepthRisk, queue_size=10)
        self.debug_pub = rospy.Publisher("/debug/depth_risk_image", Image, queue_size=2)
        rospy.Subscriber("/uav/camera/depth/image_raw", Image, self.depth_callback, queue_size=2)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback, queue_size=1)
        rospy.loginfo("depth_risk_estimator ready")

    def pose_callback(self, msg):
        self.latest_drone_z = msg.pose.position.z

    def depth_to_meters(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        encoding = depth_msg.encoding.upper()
        if encoding == "16UC1" or depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def valid_depth_mask(self, depth):
        finite = np.isfinite(depth)
        valid = np.zeros(depth.shape, dtype=bool)
        finite_values = depth[finite]
        valid[finite] = (finite_values > self.min_depth) & (finite_values < self.max_depth)
        return valid

    def q05_or_max(self, region):
        valid_mask = self.valid_depth_mask(region)
        if float(np.count_nonzero(valid_mask)) / float(region.size) < self.min_valid_ratio:
            return float(self.max_depth)
        valid = region[valid_mask]
        return float(np.percentile(valid, 5))

    def min_or_max(self, region):
        valid_mask = self.valid_depth_mask(region)
        if float(np.count_nonzero(valid_mask)) / float(region.size) < self.min_valid_ratio:
            return float(self.max_depth)
        valid = region[valid_mask]
        return float(np.min(valid))

    def front_obstacle_ratio(self, front):
        valid_mask = self.valid_depth_mask(front)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count == 0:
            return 0.0
        close_count = int(np.count_nonzero(front[valid_mask] < self.safe_depth))
        return float(close_count) / float(valid_count)

    def altitude_gate_active(self):
        return (
            self.use_altitude_gate
            and self.latest_drone_z is not None
            and self.latest_drone_z < self.min_active_height
        )

    def roi_bounds(self, height):
        y0 = int(height * self.roi_y_min_ratio)
        y1 = int(height * self.roi_y_max_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        return y0, y1

    def make_debug_image(self, depth, risk, x1, x2, y0, y1, altitude_gate):
        valid = self.valid_depth_mask(depth)
        safe_depth_image = np.where(np.isfinite(depth), depth, self.max_depth)
        clipped = np.clip(safe_depth_image, self.min_depth, self.max_depth)
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = ((1.0 - clipped[valid] / self.max_depth) * 255.0).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.line(image, (0, y0), (image.shape[1] - 1, y0), (255, 128, 0), 2)
        cv2.line(image, (0, y1 - 1), (image.shape[1] - 1, y1 - 1), (255, 128, 0), 2)
        cv2.line(image, (x1, y0), (x1, y1 - 1), (0, 255, 255), 2)
        cv2.line(image, (x2, y0), (x2, y1 - 1), (0, 255, 255), 2)
        cv2.putText(image, "L q05={:.2f}".format(risk.left_q05_depth), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "F q05={:.2f} min={:.2f}".format(risk.front_q05_depth, risk.front_min_depth), (x1 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "R q05={:.2f}".format(risk.right_q05_depth), (x2 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "area={:.2f} danger={}".format(risk.obstacle_area_ratio, risk.danger), (8, image.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if risk.danger else (0, 255, 0), 2)
        cv2.putText(image, "altitude_gate={}".format(altitude_gate), (8, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return image

    def depth_callback(self, depth_msg):
        try:
            depth = self.depth_to_meters(depth_msg)
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Depth conversion failed: %s", exc)
            return

        height, width = depth.shape[:2]
        y0, y1 = self.roi_bounds(height)
        x1 = width // 3
        x2 = 2 * width // 3
        depth_roi = depth[y0:y1, :]
        left = depth_roi[:, :x1]
        front = depth_roi[:, x1:x2]
        right = depth_roi[:, x2:]
        altitude_gate = self.altitude_gate_active()

        risk = DepthRisk()
        risk.header = depth_msg.header
        risk.left_q05_depth = self.q05_or_max(left)
        risk.front_q05_depth = self.q05_or_max(front)
        risk.right_q05_depth = self.q05_or_max(right)
        risk.front_min_depth = self.min_or_max(front)
        risk.obstacle_area_ratio = self.front_obstacle_ratio(front)
        danger_from_depth = bool(
            risk.front_q05_depth < self.danger_depth
            or risk.obstacle_area_ratio > self.danger_area_ratio
        )
        risk.danger = False if altitude_gate else danger_from_depth
        self.risk_pub.publish(risk)

        if self.debug:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(
                    self.make_debug_image(depth, risk, x1, x2, y0, y1, altitude_gate),
                    encoding="bgr8",
                )
                debug_msg.header = depth_msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as exc:
                rospy.logwarn_throttle(2.0, "Failed to publish depth debug image: %s", exc)


if __name__ == "__main__":
    try:
        DepthRiskEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
