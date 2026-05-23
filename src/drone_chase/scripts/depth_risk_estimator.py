#!/usr/bin/env python3

import json

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String

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
        self.enable_extended_debug = rospy.get_param("~enable_extended_debug", False)
        self.num_depth_sectors = max(1, int(rospy.get_param("~num_depth_sectors", 5)))
        self.temporal_smoothing_alpha = float(rospy.get_param("~temporal_smoothing_alpha", 0.3))
        self.temporal_smoothing_alpha = float(np.clip(self.temporal_smoothing_alpha, 0.0, 1.0))
        self.publish_extended_debug = rospy.get_param("~publish_extended_debug", True)
        self.use_smoothed_for_danger = rospy.get_param("~use_smoothed_for_danger", False)

        self.bridge = CvBridge()
        self.latest_drone_z = None
        self.smoothed_front_q05 = None
        self.risk_pub = rospy.Publisher("/obstacle/risk", DepthRisk, queue_size=10)
        self.debug_pub = rospy.Publisher("/debug/depth_risk_image", Image, queue_size=2)
        self.extended_debug_pub = rospy.Publisher("/debug/depth_risk_extended", String, queue_size=10)
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

    def valid_ratio(self, region):
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(self.valid_depth_mask(region))) / float(region.size)

    def sector_names(self, count):
        if int(count) == 5:
            return ["far_left", "left", "front", "right", "far_right"]
        return ["sector_{}".format(i) for i in range(int(count))]

    def sector_stats(self, region):
        valid_mask = self.valid_depth_mask(region)
        valid_count = int(np.count_nonzero(valid_mask))
        total_count = int(region.size)
        valid_ratio = float(valid_count) / float(total_count) if total_count else 0.0
        if valid_count == 0 or valid_ratio < self.min_valid_ratio:
            return {
                "q05": float(self.max_depth),
                "q10": float(self.max_depth),
                "median": float(self.max_depth),
                "valid_ratio": valid_ratio,
                "near_area_ratio": 0.0,
            }
        valid = region[valid_mask]
        near_area_ratio = float(np.count_nonzero(valid < self.safe_depth)) / float(valid_count)
        return {
            "q05": float(np.percentile(valid, 5)),
            "q10": float(np.percentile(valid, 10)),
            "median": float(np.median(valid)),
            "valid_ratio": valid_ratio,
            "near_area_ratio": near_area_ratio,
        }

    def depth_sector_stats(self, depth_roi):
        sectors = np.array_split(depth_roi, self.num_depth_sectors, axis=1)
        return {
            name: self.sector_stats(region)
            for name, region in zip(self.sector_names(len(sectors)), sectors)
        }

    def update_smoothed_front_q05(self, current_front_q05):
        current_front_q05 = float(current_front_q05)
        if self.smoothed_front_q05 is None:
            self.smoothed_front_q05 = current_front_q05
        else:
            alpha = self.temporal_smoothing_alpha
            self.smoothed_front_q05 = alpha * current_front_q05 + (1.0 - alpha) * self.smoothed_front_q05
        return float(self.smoothed_front_q05)

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

    def publish_extended_debug_msg(self, depth_msg, depth_roi, risk, smoothed_front_q05, sectors):
        if not (self.enable_extended_debug and self.publish_extended_debug):
            return
        if rospy.is_shutdown():
            return
        payload = {
            "stamp": float(depth_msg.header.stamp.to_sec()) if depth_msg.header.stamp else float(rospy.Time.now().to_sec()),
            "roi_valid_ratio": self.valid_ratio(depth_roi),
            "front_q05_raw": float(risk.front_q05_depth),
            "front_q05_smoothed": float(smoothed_front_q05),
            "sectors": sectors,
        }
        try:
            self.extended_debug_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        except rospy.ROSException:
            pass

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
        smoothed_front_q05 = self.update_smoothed_front_q05(risk.front_q05_depth)
        danger_depth_value = smoothed_front_q05 if self.use_smoothed_for_danger else risk.front_q05_depth
        danger_from_depth = bool(
            danger_depth_value < self.danger_depth
            or risk.obstacle_area_ratio > self.danger_area_ratio
        )
        risk.danger = False if altitude_gate else danger_from_depth
        if not rospy.is_shutdown():
            try:
                self.risk_pub.publish(risk)
            except rospy.ROSException:
                pass

        if self.enable_extended_debug:
            sectors = self.depth_sector_stats(depth_roi)
            self.publish_extended_debug_msg(depth_msg, depth_roi, risk, smoothed_front_q05, sectors)

        if self.debug:
            if rospy.is_shutdown():
                return
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(
                    self.make_debug_image(depth, risk, x1, x2, y0, y1, altitude_gate),
                    encoding="bgr8",
                )
                debug_msg.header = depth_msg.header
                self.debug_pub.publish(debug_msg)
            except CvBridgeError as exc:
                rospy.logwarn_throttle(2.0, "Failed to publish depth debug image: %s", exc)
            except rospy.ROSException:
                pass


if __name__ == "__main__":
    try:
        DepthRiskEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
