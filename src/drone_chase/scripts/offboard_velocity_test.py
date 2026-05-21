#!/usr/bin/env python3
"""
Phase 1: PX4 SITL + MAVROS Offboard velocity control test.

Velocity pattern: takeoff(1.5m) -> hover -> forward(1.5m) -> hover -> land
  - takeoff:  vz=0.5 m/s body-z, 3 s  -> ~1.5 m altitude
  - hover-1:  2 s
  - forward:  vx=0.3 m/s body-x, 5 s  -> ~1.5 m forward
  - hover-2:  3 s
  - land:     AUTO.LAND

Body-frame velocity -> world-frame ENU conversion before publishing.
Setpoint rate: 20 Hz.
"""

import rospy
import math
from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, ParamSet, ParamGet
from mavros_msgs.msg import ParamValue


class OffboardVelocityTest:
    def __init__(self):
        rospy.init_node("offboard_velocity_test")

        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.takeoff_speed = rospy.get_param("~takeoff_speed", 0.5)
        self.forward_speed = rospy.get_param("~forward_speed", 0.3)
        self.takeoff_dur = rospy.get_param("~takeoff_duration", 3.0)
        self.hover1_dur = rospy.get_param("~hover1_duration", 2.0)
        self.forward_dur = rospy.get_param("~forward_duration", 5.0)
        self.hover2_dur = rospy.get_param("~hover2_duration", 3.0)

        self.state = State()
        self.yaw = 0.0
        self.armed_offboard = False
        self.t0 = None

        self.vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1
        )
        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/param/set")
        rospy.wait_for_service("/mavros/param/get")
        self.arm_cli = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_cli = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo(
            "offboard_velocity_test @ %.0f Hz | takeoff %.1f m/s * %.0f s | forward %.1f m/s * %.0f s",
            self.rate_hz, self.takeoff_speed, self.takeoff_dur,
            self.forward_speed, self.forward_dur,
        )

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def body_to_enu(self, vx_b, vy_b, vz_b):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return vx_b * c - vy_b * s, vx_b * s + vy_b * c, vz_b

    def send_vel(self, vx_b, vy_b, vz_b):
        vx, vy, vz = self.body_to_enu(vx_b, vy_b, vz_b)
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self.vel_pub.publish(msg)

    def set_params_for_offboard(self):
        """Disable RC loss failsafe for SITL offboard control."""
        try:
            # COM_RCL_EXCEPT=4: allow OFFBOARD without RC
            val = ParamValue()
            val.integer = 4
            self.param_set("COM_RCL_EXCEPT", val)
            rospy.loginfo("Set COM_RCL_EXCEPT=4 (offboard without RC)")
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to set COM_RCL_EXCEPT: %s", e)
        rospy.sleep(1.0)

    def try_arm_offboard(self):
        try:
            if self.state.mode != "OFFBOARD":
                self.mode_cli(custom_mode="OFFBOARD")
            if not self.state.armed:
                self.arm_cli(True)
        except rospy.ServiceException:
            pass
        if self.state.armed and self.state.mode == "OFFBOARD":
            if not self.armed_offboard:
                rospy.loginfo("ARMED + OFFBOARD!")
                self.armed_offboard = True
                self.t0 = rospy.get_time()

    def run(self):
        # Stream initial setpoints for 5 s (PX4 requirement)
        rospy.loginfo("Streaming initial setpoints (5 s)...")
        for _ in range(int(self.rate_hz * 5)):
            self.send_vel(0, 0, 0)
            self.rate.sleep()

        # Wait for FCU
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.state.connected:
            self.send_vel(0, 0, 0)
            self.rate.sleep()
        rospy.loginfo("FCU connected.")

        # Disable RC failsafe for SITL
        self.set_params_for_offboard()

        # Arm + OFFBOARD
        rospy.loginfo("Requesting ARM + OFFBOARD...")
        while not rospy.is_shutdown() and not self.armed_offboard:
            self.send_vel(0, 0, self.takeoff_speed)
            self.try_arm_offboard()
            self.rate.sleep()

        # Velocity pattern
        t1 = self.takeoff_dur
        t2 = t1 + self.hover1_dur
        t3 = t2 + self.forward_dur
        t4 = t3 + self.hover2_dur

        rospy.loginfo("Executing pattern...")
        while not rospy.is_shutdown():
            t = rospy.get_time() - self.t0

            if t < t1:
                rospy.loginfo_throttle(
                    1.0, "TAKEOFF  t=%.1f  vz=%.1f", t, self.takeoff_speed)
                self.send_vel(0, 0, self.takeoff_speed)

            elif t < t2:
                rospy.loginfo_throttle(1.0, "HOVER-1  t=%.1f", t)
                self.send_vel(0, 0, 0)

            elif t < t3:
                rospy.loginfo_throttle(
                    1.0, "FORWARD  t=%.1f  vx=%.1f", t, self.forward_speed)
                self.send_vel(self.forward_speed, 0, 0)

            elif t < t4:
                rospy.loginfo_throttle(1.0, "HOVER-2  t=%.1f", t)
                self.send_vel(0, 0, 0)

            else:
                rospy.loginfo("Pattern complete, AUTO.LAND")
                self.mode_cli(custom_mode="AUTO.LAND")
                return

            self.try_arm_offboard()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = OffboardVelocityTest()
        node.run()
    except rospy.ROSInterruptException:
        pass
