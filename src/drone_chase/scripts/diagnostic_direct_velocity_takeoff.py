#!/usr/bin/env python3
"""Direct MAVROS velocity takeoff diagnostic.

This script intentionally bypasses /raw_cmd_vel, safety_filter_node, and
GazeboChaseEnv. It only verifies whether PX4 SITL + MAVROS velocity setpoints
can produce lift in the currently loaded Gazebo model.
"""

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, RCOut, State
from mavros_msgs.srv import CommandBool, ParamSet, SetMode


class DirectVelocityTakeoff:
    def __init__(self):
        rospy.init_node("diagnostic_direct_velocity_takeoff")

        self.takeoff_duration = float(rospy.get_param("~takeoff_duration", 5.0))
        self.hover_duration = float(rospy.get_param("~hover_duration", 3.0))
        self.takeoff_vz = float(rospy.get_param("~takeoff_vz", 0.5))

        self.state = State()
        self.pose = None
        self.model_states = None
        self.rc_out = None

        self.vel_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1
        )
        rospy.Subscriber("/mavros/state", State, self._state_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb)
        rospy.Subscriber("/mavros/rc/out", RCOut, self._rc_out_cb)

        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/param/set")

        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.rate = rospy.Rate(20.0)

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def _model_states_cb(self, msg):
        self.model_states = msg

    def _rc_out_cb(self, msg):
        self.rc_out = msg

    def z(self):
        return float("nan") if self.pose is None else self.pose.pose.position.z

    def gazebo_z(self):
        if self.model_states is None:
            return float("nan")
        for name, pose in zip(self.model_states.name, self.model_states.pose):
            if name in ("iris", "iris_rgbd_chaser"):
                return pose.position.z
        return float("nan")

    def rc_summary(self):
        if self.rc_out is None:
            return []
        return list(self.rc_out.channels[:4])

    def send(self, z_vel):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = z_vel
        msg.twist.angular.z = 0.0
        self.vel_pub.publish(msg)

    def allow_offboard_without_rc(self):
        value = ParamValue()
        value.integer = 4
        value.real = 0.0
        try:
            self.param_set("COM_RCL_EXCEPT", value)
            rospy.loginfo("Set COM_RCL_EXCEPT=4")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to set COM_RCL_EXCEPT=4: %s", exc)

    def run(self):
        rospy.loginfo("Waiting for FCU connection...")
        while not rospy.is_shutdown() and not self.state.connected:
            self.send(0.0)
            self.rate.sleep()
        rospy.loginfo("FCU connected, initial local_z=%.3f gazebo_z=%.3f", self.z(), self.gazebo_z())

        self.allow_offboard_without_rc()

        rospy.loginfo("Sending 100 zero setpoints...")
        for _ in range(100):
            if rospy.is_shutdown():
                return
            self.send(0.0)
            self.rate.sleep()

        rospy.loginfo("Requesting OFFBOARD + arm...")
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            self.send(0.5)
            if self.state.mode != "OFFBOARD":
                try:
                    self.set_mode(custom_mode="OFFBOARD")
                except rospy.ServiceException:
                    pass
            if not self.state.armed:
                try:
                    self.arm(True)
                except rospy.ServiceException:
                    pass
            if self.state.mode == "OFFBOARD" and self.state.armed:
                break
            if (rospy.Time.now() - start).to_sec() > 8.0:
                rospy.logwarn(
                    "Timed out before OFFBOARD+armed: connected=%s mode=%s armed=%s local_z=%.3f gazebo_z=%.3f rc=%s",
                    self.state.connected,
                    self.state.mode,
                    self.state.armed,
                    self.z(),
                    self.gazebo_z(),
                    self.rc_summary(),
                )
                return
            self.rate.sleep()

        rospy.loginfo(
            "OFFBOARD+armed, takeoff command z_vel=%.2f for %.1fs",
            self.takeoff_vz,
            self.takeoff_duration,
        )
        takeoff_start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - takeoff_start).to_sec() < self.takeoff_duration:
            self.send(self.takeoff_vz)
            rospy.loginfo_throttle(
                1.0,
                "TAKEOFF local_z=%.3f gazebo_z=%.3f mode=%s armed=%s rc=%s",
                self.z(),
                self.gazebo_z(),
                self.state.mode,
                self.state.armed,
                self.rc_summary(),
            )
            self.rate.sleep()

        rospy.loginfo("Hover command z_vel=0 for %.1fs", self.hover_duration)
        hover_start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - hover_start).to_sec() < self.hover_duration:
            self.send(0.0)
            rospy.loginfo_throttle(
                1.0,
                "HOVER local_z=%.3f gazebo_z=%.3f mode=%s armed=%s rc=%s",
                self.z(),
                self.gazebo_z(),
                self.state.mode,
                self.state.armed,
                self.rc_summary(),
            )
            self.rate.sleep()

        rospy.loginfo(
            "Final local_z=%.3f gazebo_z=%.3f mode=%s armed=%s rc=%s",
            self.z(),
            self.gazebo_z(),
            self.state.mode,
            self.state.armed,
            self.rc_summary(),
        )


if __name__ == "__main__":
    try:
        DirectVelocityTakeoff().run()
    except rospy.ROSInterruptException:
        pass
