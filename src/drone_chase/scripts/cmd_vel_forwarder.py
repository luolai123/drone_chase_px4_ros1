#!/usr/bin/env python3
"""Forward body-frame /raw_cmd_vel commands to MAVROS setpoint velocity."""

import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, State
from mavros_msgs.srv import CommandBool, ParamSet, SetMode


class CmdVelForwarder:
    STATE_WAIT_FCU = "WAIT_FCU"
    STATE_PRESTREAM = "PRESTREAM"
    STATE_OFFBOARD_ARM = "OFFBOARD_ARM"
    STATE_TAKEOFF = "TAKEOFF"
    STATE_FORWARDING = "FORWARDING"

    def __init__(self):
        rospy.init_node("cmd_vel_forwarder")

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.target_altitude = float(rospy.get_param("~target_altitude", 1.5))
        self.takeoff_vz = float(rospy.get_param("~takeoff_vz", 0.5))
        self.takeoff_tolerance = float(rospy.get_param("~takeoff_tolerance", 0.1))
        self.raw_timeout = float(rospy.get_param("~raw_timeout", 0.5))
        self.service_retry_interval = float(rospy.get_param("~service_retry_interval", 1.0))
        self.set_com_rcl_except = bool(rospy.get_param("~set_com_rcl_except", True))
        self.debug = bool(rospy.get_param("~debug", True))

        self.state = State()
        self.pose = None
        self.raw_cmd = None
        self.raw_stamp = None
        self.forwarder_state = self.STATE_WAIT_FCU
        self.last_service_request = rospy.Time(0)

        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel",
            TwistStamped,
            queue_size=10,
        )
        rospy.Subscriber("/raw_cmd_vel", TwistStamped, self.raw_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=1)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/param/set")
        self.arm_cli = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_cli = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.param_set_cli = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("cmd_vel_forwarder ready")

    def raw_cb(self, msg):
        self.raw_cmd = msg
        self.raw_stamp = rospy.Time.now()

    def pose_cb(self, msg):
        self.pose = msg

    def state_cb(self, msg):
        self.state = msg

    def current_z(self):
        if self.pose is None:
            return 0.0
        return self.pose.pose.position.z

    def current_yaw(self):
        if self.pose is None:
            rospy.logwarn_throttle(2.0, "No /mavros/local_position/pose yet; assuming yaw=0")
            return 0.0
        q = self.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def raw_age(self):
        if self.raw_stamp is None:
            return math.inf
        return (rospy.Time.now() - self.raw_stamp).to_sec()

    def fresh_raw_cmd(self):
        if self.raw_cmd is None or self.raw_age() > self.raw_timeout:
            return None
        return self.raw_cmd

    def service_request_due(self):
        now = rospy.Time.now()
        return (now - self.last_service_request).to_sec() >= self.service_retry_interval

    def request_offboard_and_arm(self):
        if not self.service_request_due():
            return
        self.last_service_request = rospy.Time.now()

        try:
            if self.state.mode != "OFFBOARD":
                self.mode_cli(custom_mode="OFFBOARD")
            if not self.state.armed:
                self.arm_cli(True)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "OFFBOARD/arming request failed: %s", exc)

    def configure_offboard_params(self):
        if not self.set_com_rcl_except:
            return
        try:
            value = ParamValue()
            value.integer = 4
            response = self.param_set_cli("COM_RCL_EXCEPT", value)
            if response.success:
                rospy.loginfo("Set COM_RCL_EXCEPT=4 for no-RC SITL OFFBOARD")
            else:
                rospy.logwarn("COM_RCL_EXCEPT set request returned success=False")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to set COM_RCL_EXCEPT: %s", exc)

    def body_to_world(self, vx_body, vy_body, vz_body, yaw_rate):
        yaw = self.current_yaw()
        vx_world = math.cos(yaw) * vx_body - math.sin(yaw) * vy_body
        vy_world = math.sin(yaw) * vx_body + math.cos(yaw) * vy_body
        return vx_world, vy_world, vz_body, yaw_rate

    def publish_world_velocity(self, vx_world, vy_world, vz_world, yaw_rate):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.twist.linear.x = float(vx_world)
        msg.twist.linear.y = float(vy_world)
        msg.twist.linear.z = float(vz_world)
        msg.twist.angular.z = float(yaw_rate)
        self.setpoint_pub.publish(msg)
        return msg

    def publish_zero(self):
        return self.publish_world_velocity(0.0, 0.0, 0.0, 0.0)

    def publish_takeoff(self):
        if self.current_z() < self.target_altitude - self.takeoff_tolerance:
            return self.publish_world_velocity(0.0, 0.0, self.takeoff_vz, 0.0)
        self.forwarder_state = self.STATE_FORWARDING
        rospy.loginfo("cmd_vel_forwarder reached takeoff altitude %.2f m", self.current_z())
        return self.publish_zero()

    def publish_forwarded_raw(self):
        raw = self.fresh_raw_cmd()
        if raw is None:
            return self.publish_zero()

        vx_world, vy_world, vz_world, yaw_rate = self.body_to_world(
            raw.twist.linear.x,
            raw.twist.linear.y,
            raw.twist.linear.z,
            raw.twist.angular.z,
        )
        return self.publish_world_velocity(vx_world, vy_world, vz_world, yaw_rate)

    def log_status(self, published_msg):
        if not self.debug:
            return
        age = self.raw_age()
        if math.isinf(age):
            age = -1.0
        rospy.loginfo_throttle(
            1.0,
            "connected=%s armed=%s mode=%s current_z=%.2f forwarder_state=%s "
            "raw_age=%.2f pub[vx=%.2f vy=%.2f vz=%.2f yaw=%.2f]",
            self.state.connected,
            self.state.armed,
            self.state.mode,
            self.current_z(),
            self.forwarder_state,
            age,
            published_msg.twist.linear.x,
            published_msg.twist.linear.y,
            published_msg.twist.linear.z,
            published_msg.twist.angular.z,
        )

    def prestream_setpoints(self):
        self.forwarder_state = self.STATE_PRESTREAM
        rospy.loginfo("Streaming 100 zero setpoints before OFFBOARD")
        for _ in range(100):
            if rospy.is_shutdown():
                return
            msg = self.publish_zero()
            self.log_status(msg)
            self.rate.sleep()

    def wait_for_fcu(self):
        self.forwarder_state = self.STATE_WAIT_FCU
        rospy.loginfo("Waiting for MAVROS FCU connection")
        while not rospy.is_shutdown() and not self.state.connected:
            msg = self.publish_zero()
            self.log_status(msg)
            self.rate.sleep()
        rospy.loginfo("MAVROS FCU connected")

    def ensure_offboard_armed(self):
        self.forwarder_state = self.STATE_OFFBOARD_ARM
        while not rospy.is_shutdown() and (self.state.mode != "OFFBOARD" or not self.state.armed):
            msg = self.publish_zero()
            self.request_offboard_and_arm()
            self.log_status(msg)
            self.rate.sleep()
        rospy.loginfo("cmd_vel_forwarder OFFBOARD + armed")

    def run(self):
        self.wait_for_fcu()
        self.configure_offboard_params()
        self.prestream_setpoints()
        self.ensure_offboard_armed()
        self.forwarder_state = self.STATE_TAKEOFF

        while not rospy.is_shutdown():
            if self.state.mode != "OFFBOARD" or not self.state.armed:
                self.request_offboard_and_arm()

            if self.forwarder_state == self.STATE_TAKEOFF:
                msg = self.publish_takeoff()
            else:
                msg = self.publish_forwarded_raw()
            self.log_status(msg)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        CmdVelForwarder().run()
    except rospy.ROSInterruptException:
        pass
