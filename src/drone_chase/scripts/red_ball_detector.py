#!/usr/bin/env python3

import json
import math
from collections import deque

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from drone_chase.msg import TargetState


class RedBallDetector:
    def __init__(self):
        rospy.init_node("red_ball_detector")

        self.min_area = rospy.get_param("~min_area", 30.0)
        self.min_circularity = rospy.get_param("~min_circularity", 0.4)
        self.min_radius_px = rospy.get_param("~min_radius_px", 2.5)
        self.min_depth = rospy.get_param("~min_depth", 0.2)
        self.max_depth = rospy.get_param("~max_depth", 10.0)
        self.debug = rospy.get_param("~debug", True)
        self.enable_detection_quality_debug = rospy.get_param("~enable_detection_quality_debug", False)
        self.publish_detection_quality = rospy.get_param("~publish_detection_quality", True)
        self.target_lost_window = max(1, int(rospy.get_param("~target_lost_window", 20)))
        self.min_valid_depth_ratio = rospy.get_param(
            "~min_valid_depth_ratio",
            rospy.get_param("~min_depth_valid_ratio", 0.2),
        )
        queue_size = rospy.get_param("~sync_queue_size", 10)
        slop = rospy.get_param("~sync_slop", 0.12)

        self.bridge = CvBridge()
        self.latest_pose = None
        self.visible_history = deque(maxlen=self.target_lost_window)
        self.lost_frames = 0
        self.visible_frames = 0
        self.radius_px_smooth = 0.0
        self.radius_smoothing_alpha = 0.3

        self.state_pub = rospy.Publisher("/target/state", TargetState, queue_size=10)
        self.mask_pub = rospy.Publisher("/debug/red_ball_mask", Image, queue_size=2)
        self.overlay_pub = rospy.Publisher("/debug/red_ball_overlay", Image, queue_size=2)
        self.quality_pub = rospy.Publisher("/debug/target_detection_quality", String, queue_size=10)

        rgb_sub = message_filters.Subscriber("/uav/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/uav/camera/depth/image_raw", Image)
        info_sub = message_filters.Subscriber("/uav/camera/rgb/camera_info", CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=queue_size, slop=slop
        )
        self.sync.registerCallback(self.image_callback)

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback, queue_size=1)
        rospy.loginfo("red_ball_detector ready")

    def pose_callback(self, msg):
        self.latest_pose = msg

    def depth_to_meters(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        encoding = depth_msg.encoding.upper()
        if encoding == "16UC1" or depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)
        return depth_m

    def make_empty_state(self, header):
        msg = TargetState()
        msg.header = header
        msg.visible = False
        msg.u = 0.0
        msg.v = 0.0
        msg.radius_px = 0.0
        msg.depth = float(self.max_depth)
        msg.position_camera = Point()
        msg.position_world = Point()
        msg.confidence = 0.0
        return msg

    def find_red_ball(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 80, 80], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if area <= self.min_area or circularity <= self.min_circularity or radius <= self.min_radius_px:
                continue
            if best is None or area > best["area"]:
                best = {
                    "contour": contour,
                    "area": area,
                    "circularity": circularity,
                    "center": (float(cx), float(cy)),
                    "radius": float(radius),
                }
        return mask, best

    def estimate_target_depth(self, depth_m, contour, center, radius):
        target_mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
        cv2.drawContours(target_mask, [contour], -1, 255, thickness=-1)
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(target_mask, (cx, cy), max(1, int(round(radius * 0.65))), 255, thickness=-1)

        candidate = depth_m[target_mask > 0]
        finite = np.isfinite(candidate)
        finite_values = candidate[finite]
        valid = finite_values[
            (finite_values > self.min_depth)
            & (finite_values < self.max_depth)
        ]
        total_pixels = max(1, int(np.count_nonzero(target_mask)))
        valid_ratio = float(valid.size) / float(total_pixels)
        if valid.size == 0 or valid_ratio < self.min_valid_depth_ratio:
            return None, valid_ratio
        return float(np.median(valid)), valid_ratio

    def update_detection_quality_state(self, visible, radius_px):
        self.visible_history.append(bool(visible))
        if visible:
            self.visible_frames += 1
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            self.visible_frames = 0

        radius_px = float(radius_px)
        if visible:
            if self.radius_px_smooth <= 0.0:
                self.radius_px_smooth = radius_px
            else:
                alpha = self.radius_smoothing_alpha
                self.radius_px_smooth = alpha * radius_px + (1.0 - alpha) * self.radius_px_smooth
        return float(sum(1 for item in self.visible_history if item)) / float(len(self.visible_history))

    def classify_detection_quality(self, state, target, depth_valid_ratio, visible_ratio_window):
        if not state.visible:
            return "unstable" if visible_ratio_window > 0.0 and self.lost_frames < self.target_lost_window else "lost"
        if depth_valid_ratio < max(0.5, float(self.min_valid_depth_ratio)):
            return "weak_depth"
        if state.radius_px < max(float(self.min_radius_px) * 1.5, 4.0):
            return "small_target"
        if visible_ratio_window < 0.8 or self.lost_frames > 0:
            return "unstable"
        return "good"

    def publish_detection_quality_msg(self, state, target, depth_valid_ratio):
        if not (self.enable_detection_quality_debug and self.publish_detection_quality):
            return
        if rospy.is_shutdown():
            return
        visible_ratio_window = self.update_detection_quality_state(state.visible, state.radius_px)
        quality = self.classify_detection_quality(state, target, depth_valid_ratio, visible_ratio_window)
        payload = {
            "visible": bool(state.visible),
            "visible_ratio_window": float(visible_ratio_window),
            "lost_frames": int(self.lost_frames),
            "visible_frames": int(self.visible_frames),
            "depth_valid_ratio": float(depth_valid_ratio),
            "radius_px": float(state.radius_px),
            "radius_px_smooth": float(self.radius_px_smooth),
            "confidence": float(state.confidence),
            "quality": quality,
        }
        try:
            self.quality_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        except rospy.ROSException:
            pass

    def camera_to_world_approx(self, position_camera):
        world = Point()
        if self.latest_pose is None:
            return world

        drone = self.latest_pose.pose.position
        # Phase 3 approximation only: camera axes are mapped into MAVROS local ENU
        # without using vehicle yaw/TF. Later phases should replace this with tf2.
        world.x = drone.x + position_camera.z
        world.y = drone.y - position_camera.x
        world.z = drone.z - position_camera.y
        return world

    def publish_debug(self, header, bgr, mask, target, state):
        if not self.debug:
            return
        if rospy.is_shutdown():
            return

        overlay = bgr.copy()
        if target is not None and state.visible:
            contour = target["contour"]
            cx, cy = target["center"]
            radius = target["radius"]
            center_i = (int(round(cx)), int(round(cy)))
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
            cv2.circle(overlay, center_i, int(round(radius)), (255, 0, 0), 2)
            cv2.circle(overlay, center_i, 3, (0, 255, 255), -1)
            text = "Z={:.2f}m u={:.2f} v={:.2f} conf={:.2f}".format(
                state.depth, state.u, state.v, state.confidence
            )
            cv2.putText(overlay, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            cv2.putText(overlay, "visible=False", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = header
            self.mask_pub.publish(mask_msg)

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = header
            self.overlay_pub.publish(overlay_msg)
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Failed to publish red-ball debug images: %s", exc)
        except rospy.ROSException:
            pass

    def image_callback(self, rgb_msg, depth_msg, info_msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_m = self.depth_to_meters(depth_msg)
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Image conversion failed: %s", exc)
            return

        header = rgb_msg.header
        state = self.make_empty_state(header)
        mask, target = self.find_red_ball(bgr)
        depth_valid_ratio = 0.0

        if target is not None:
            depth, valid_ratio = self.estimate_target_depth(
                depth_m, target["contour"], target["center"], target["radius"]
            )
            depth_valid_ratio = float(valid_ratio)
            fx, fy = info_msg.K[0], info_msg.K[4]
            cx0, cy0 = info_msg.K[2], info_msg.K[5]
            if depth is not None and fx > 0.0 and fy > 0.0:
                height, width = bgr.shape[:2]
                center_x, center_y = target["center"]
                position_camera = Point()
                position_camera.x = (center_x - cx0) * depth / fx
                position_camera.y = (center_y - cy0) * depth / fy
                position_camera.z = depth

                area_score = min(1.0, target["area"] / float(width * height) * 30.0)
                circularity_score = min(1.0, max(0.0, target["circularity"]))
                depth_score = min(1.0, max(0.0, valid_ratio / 0.5))

                state.visible = True
                state.u = float((center_x - width / 2.0) / (width / 2.0))
                state.v = float((center_y - height / 2.0) / (height / 2.0))
                state.radius_px = target["radius"]
                state.depth = depth
                state.position_camera = position_camera
                state.position_world = self.camera_to_world_approx(position_camera)
                state.confidence = float(np.clip(0.4 * area_score + 0.4 * circularity_score + 0.2 * depth_score, 0.0, 1.0))
            else:
                rospy.logwarn_throttle(2.0, "Red contour found but depth/CameraInfo is not usable")

        if not rospy.is_shutdown():
            try:
                self.state_pub.publish(state)
            except rospy.ROSException:
                pass
        self.publish_detection_quality_msg(state, target, depth_valid_ratio)
        self.publish_debug(header, bgr, mask, target, state)


if __name__ == "__main__":
    try:
        RedBallDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
