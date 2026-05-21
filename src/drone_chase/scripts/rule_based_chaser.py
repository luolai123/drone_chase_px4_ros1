#!/usr/bin/env python3
"""Phase 4 rule-based red ball chaser.

Publishes body-frame semantic velocity commands on /raw_cmd_vel. It does not
publish MAVROS setpoints directly.
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State

from drone_chase.msg import DepthRisk, TargetState


def clamp(value, low, high):
    return max(low, min(high, value))


class RuleBasedChaser:
    MODE_SEARCH = "SEARCH"
    MODE_TRACK = "TRACK"
    MODE_CAPTURED = "CAPTURED"
    MODE_AVOID = "AVOID"

    def __init__(self):
        rospy.init_node("rule_based_chaser")

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.desired_distance = float(rospy.get_param("~desired_distance", 0.8))
        self.capture_distance = float(rospy.get_param("~capture_distance", 0.5))

        self.k_dist = float(rospy.get_param("~k_dist", 0.35))
        self.k_yaw = float(rospy.get_param("~k_yaw", 0.8))
        self.k_z = float(rospy.get_param("~k_z", 0.35))

        self.max_vx = float(rospy.get_param("~max_vx", 0.5))
        self.max_vy = float(rospy.get_param("~max_vy", 0.3))
        self.max_vz = float(rospy.get_param("~max_vz", 0.25))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.6))

        self.min_vx = float(rospy.get_param("~min_vx", -0.2))
        self.search_yaw_rate = float(rospy.get_param("~search_yaw_rate", 0.3))

        self.avoid_stop_depth = float(rospy.get_param("~avoid_stop_depth", 0.6))
        self.avoid_emergency_depth = float(rospy.get_param("~avoid_emergency_depth", 0.35))

        self.min_height = float(rospy.get_param("~min_height", 0.6))
        self.max_height = float(rospy.get_param("~max_height", 2.5))
        self.data_timeout = float(rospy.get_param("~data_timeout", 0.5))
        self.debug = bool(rospy.get_param("~debug", True))

        self.target = None
        self.target_stamp = None
        self.risk = None
        self.risk_stamp = None
        self.pose = None
        self.pose_stamp = None
        self.velocity = None
        self.state = State()

        self.cmd_pub = rospy.Publisher("/raw_cmd_vel", TwistStamped, queue_size=10)
        rospy.Subscriber("/target/state", TargetState, self.target_cb, queue_size=1)
        rospy.Subscriber("/obstacle/risk", DepthRisk, self.risk_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=1)

        rospy.loginfo("rule_based_chaser ready")

    def target_cb(self, msg):
        self.target = msg
        self.target_stamp = rospy.Time.now()

    def risk_cb(self, msg):
        self.risk = msg
        self.risk_stamp = rospy.Time.now()

    def pose_cb(self, msg):
        self.pose = msg
        self.pose_stamp = rospy.Time.now()

    def velocity_cb(self, msg):
        self.velocity = msg

    def state_cb(self, msg):
        self.state = msg

    def is_fresh(self, stamp):
        if stamp is None:
            return False
        return (rospy.Time.now() - stamp).to_sec() <= self.data_timeout

    def fresh_target(self):
        return self.target if self.is_fresh(self.target_stamp) else None

    def fresh_risk(self):
        return self.risk if self.is_fresh(self.risk_stamp) else None

    def fresh_pose(self):
        return self.pose if self.is_fresh(self.pose_stamp) else None

    def base_command(self, target):
        if target is None or not target.visible:
            return 0.0, 0.0, 0.0, self.search_yaw_rate, self.MODE_SEARCH

        yaw_rate = -self.k_yaw * target.u
        vz = -self.k_z * target.v
        distance_error = target.depth - self.desired_distance
        vx = self.k_dist * distance_error
        vy = 0.0
        mode = self.MODE_TRACK

        vx = clamp(vx, self.min_vx, self.max_vx)
        vy = clamp(vy, -self.max_vy, self.max_vy)
        vz = clamp(vz, -self.max_vz, self.max_vz)
        yaw_rate = clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        if target.depth < self.capture_distance:
            vx = 0.0
            vy = 0.0
            vz = 0.0
            yaw_rate = 0.0
            mode = self.MODE_CAPTURED

        return vx, vy, vz, yaw_rate, mode

    def apply_obstacle_override(self, vx, vy, vz, yaw_rate, mode):
        risk = self.fresh_risk()
        if risk is None:
            rospy.logwarn_throttle(2.0, "No fresh /obstacle/risk; obstacle override disabled")
            return vx, vy, vz, yaw_rate, mode

        if risk.front_q05_depth < self.avoid_stop_depth:
            vx = min(vx, 0.0)
            mode = self.MODE_AVOID

        emergency = (
            risk.front_q05_depth < self.avoid_emergency_depth
            or (risk.danger and risk.obstacle_area_ratio > 0.5)
        )
        if emergency:
            vx = -0.2
            vy = 0.0
            vz = 0.1
            yaw_rate = 0.3
            mode = self.MODE_AVOID

        return vx, vy, vz, yaw_rate, mode

    def apply_height_guard(self, vz):
        pose = self.fresh_pose()
        if pose is None:
            rospy.logwarn_throttle(2.0, "No fresh /mavros/local_position/pose; height guard disabled")
            return vz

        z = pose.pose.position.z
        if z < self.min_height:
            vz = max(vz, 0.0)
        if z > self.max_height:
            vz = min(vz, 0.0)
        return vz

    def publish_command(self, vx, vy, vz, yaw_rate):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.z = float(yaw_rate)
        self.cmd_pub.publish(msg)

    def log_status(self, mode, target, risk, vx, vy, vz, yaw_rate):
        if not self.debug:
            return

        visible = bool(target.visible) if target is not None else False
        target_depth = target.depth if target is not None else math.nan
        target_u = target.u if target is not None else 0.0
        target_v = target.v if target is not None else 0.0
        front_q05 = risk.front_q05_depth if risk is not None else math.nan
        area_ratio = risk.obstacle_area_ratio if risk is not None else math.nan
        danger = bool(risk.danger) if risk is not None else False
        pose = self.fresh_pose()
        drone_z = pose.pose.position.z if pose is not None else math.nan

        rospy.loginfo_throttle(
            1.0,
            "mode=%s target_visible=%s target_depth=%.2f u=%.2f v=%.2f "
            "front_q05=%.2f area=%.2f danger=%s cmd[vx=%.2f vy=%.2f vz=%.2f yaw=%.2f] drone_z=%.2f",
            mode,
            visible,
            target_depth,
            target_u,
            target_v,
            front_q05,
            area_ratio,
            danger,
            vx,
            vy,
            vz,
            yaw_rate,
            drone_z,
        )

    def step(self):
        target = self.fresh_target()
        risk = self.fresh_risk()
        vx, vy, vz, yaw_rate, mode = self.base_command(target)
        vx, vy, vz, yaw_rate, mode = self.apply_obstacle_override(vx, vy, vz, yaw_rate, mode)
        vz = self.apply_height_guard(vz)
        self.publish_command(vx, vy, vz, yaw_rate)
        self.log_status(mode, target, risk, vx, vy, vz, yaw_rate)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()


if __name__ == "__main__":
    try:
        RuleBasedChaser().run()
    except rospy.ROSInterruptException:
        pass
