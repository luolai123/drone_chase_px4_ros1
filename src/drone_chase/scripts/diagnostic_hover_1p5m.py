#!/usr/bin/env python3
"""
Diagnostic only: take off to about 1.5 m and hover for Phase 3.7 risk checks.

This node publishes /mavros/setpoint_velocity/cmd_vel at 20 Hz. It does not
publish /raw_cmd_vel and does not implement chase behavior.
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, State
from mavros_msgs.srv import CommandBool, ParamSet, SetMode


class DiagnosticHover:
    def __init__(self):
        rospy.init_node("diagnostic_hover_1p5m")

        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.target_z = rospy.get_param("~target_z", 1.5)
        self.hover_duration = rospy.get_param("~hover_duration", 30.0)
        self.kp_z = rospy.get_param("~kp_z", 0.8)
        self.kp_xy = rospy.get_param("~kp_xy", 0.35)
        self.max_up_speed = rospy.get_param("~max_up_speed", 0.6)
        self.max_down_speed = rospy.get_param("~max_down_speed", 0.35)
        self.max_xy_speed = rospy.get_param("~max_xy_speed", 0.25)
        self.reached_tolerance = rospy.get_param("~reached_tolerance", 0.12)

        self.state = State()
        self.pose = None
        self.hold_x = None
        self.hold_y = None
        self.armed_offboard = False

        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=1)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/param/set")
        self.arm_cli = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_cli = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.rate = rospy.Rate(self.rate_hz)

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.pose = msg

    def send_velocity(self, vx=0.0, vy=0.0, vz=0.0):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        self.vel_pub.publish(msg)

    def set_offboard_params(self):
        try:
            val = ParamValue()
            val.integer = 4
            self.param_set("COM_RCL_EXCEPT", val)
            rospy.loginfo("Set COM_RCL_EXCEPT=4 for diagnostic offboard hover")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to set COM_RCL_EXCEPT: %s", exc)

    def try_arm_offboard(self):
        try:
            if self.state.mode != "OFFBOARD":
                self.mode_cli(custom_mode="OFFBOARD")
            if not self.state.armed:
                self.arm_cli(True)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "ARM/OFFBOARD request failed: %s", exc)

        if self.state.armed and self.state.mode == "OFFBOARD" and not self.armed_offboard:
            self.armed_offboard = True
            rospy.loginfo("Diagnostic hover ARMED + OFFBOARD")

    def current_z(self):
        if self.pose is None:
            return 0.0
        return self.pose.pose.position.z

    def current_x(self):
        if self.pose is None:
            return 0.0
        return self.pose.pose.position.x

    def current_y(self):
        if self.pose is None:
            return 0.0
        return self.pose.pose.position.y

    def wait_for_pose(self):
        while not rospy.is_shutdown() and self.pose is None:
            self.send_velocity()
            self.rate.sleep()

    def capture_hold_xy(self):
        if self.pose is None:
            return
        self.hold_x = self.pose.pose.position.x
        self.hold_y = self.pose.pose.position.y
        rospy.loginfo("Diagnostic hover holding XY at x=%.2f y=%.2f", self.hold_x, self.hold_y)

    def z_velocity_command(self):
        err = self.target_z - self.current_z()
        cmd = self.kp_z * err
        return max(-self.max_down_speed, min(self.max_up_speed, cmd))

    def xy_velocity_command(self):
        if self.pose is None or self.hold_x is None or self.hold_y is None:
            return 0.0, 0.0
        vx = self.kp_xy * (self.hold_x - self.current_x())
        vy = self.kp_xy * (self.hold_y - self.current_y())
        vx = max(-self.max_xy_speed, min(self.max_xy_speed, vx))
        vy = max(-self.max_xy_speed, min(self.max_xy_speed, vy))
        return vx, vy

    def send_hold_velocity(self, vz=None):
        vx, vy = self.xy_velocity_command()
        if vz is None:
            vz = self.z_velocity_command()
        self.send_velocity(vx, vy, vz)

    def stream_initial_setpoints(self):
        rospy.loginfo("Streaming initial diagnostic setpoints")
        for _ in range(int(self.rate_hz * 5.0)):
            if rospy.is_shutdown():
                return
            self.send_velocity()
            self.rate.sleep()

    def run(self):
        self.stream_initial_setpoints()

        rospy.loginfo("Waiting for FCU connection")
        while not rospy.is_shutdown() and not self.state.connected:
            self.send_velocity()
            self.rate.sleep()
        rospy.loginfo("FCU connected")
        self.wait_for_pose()
        self.capture_hold_xy()

        self.set_offboard_params()

        while not rospy.is_shutdown() and not self.armed_offboard:
            self.send_hold_velocity(self.max_up_speed * 0.5)
            self.try_arm_offboard()
            self.rate.sleep()

        rospy.loginfo("Climbing to %.2f m", self.target_z)
        stable_count = 0
        while not rospy.is_shutdown():
            z_err = self.target_z - self.current_z()
            self.send_hold_velocity()
            self.try_arm_offboard()

            if math.fabs(z_err) < self.reached_tolerance:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= int(self.rate_hz * 1.0):
                break
            self.rate.sleep()

        rospy.loginfo("Hovering for %.1f s at z=%.2f", self.hover_duration, self.current_z())
        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < self.hover_duration:
            self.send_hold_velocity()
            self.try_arm_offboard()
            rospy.loginfo_throttle(
                2.0,
                "Diagnostic hover x=%.2f y=%.2f z=%.2f target_z=%.2f",
                self.current_x(),
                self.current_y(),
                self.current_z(),
                self.target_z,
            )
            self.rate.sleep()

        rospy.loginfo("Diagnostic hover complete; holding current altitude until shutdown")
        while not rospy.is_shutdown():
            self.send_hold_velocity()
            self.try_arm_offboard()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        DiagnosticHover().run()
    except rospy.ROSInterruptException:
        pass
