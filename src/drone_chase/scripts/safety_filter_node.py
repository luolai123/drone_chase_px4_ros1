#!/usr/bin/env python3
"""Phase 5 safety filter for policy velocity commands.

Consumes body-frame semantic commands on /raw_cmd_vel, applies common safety
guards, converts the result to MAVROS ENU setpoint velocity, and owns the
OFFBOARD/arming/takeoff lifecycle.
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, State
from mavros_msgs.srv import CommandBool, ParamSet, SetMode
from std_msgs.msg import String

from drone_chase.msg import DepthRisk, TargetState


def clamp(value, low, high):
    return max(low, min(high, value))


def finite(value):
    return math.isfinite(float(value))


class BodyCommand:
    __slots__ = ("vx", "vy", "vz", "yaw_rate")

    def __init__(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.yaw_rate = float(yaw_rate)

    @classmethod
    def from_msg(cls, msg):
        return cls(
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.z,
        )

    def all_finite(self):
        return (
            finite(self.vx)
            and finite(self.vy)
            and finite(self.vz)
            and finite(self.yaw_rate)
        )

    def copy(self):
        return BodyCommand(self.vx, self.vy, self.vz, self.yaw_rate)


class SafetyFilterNode:
    STATE_WAIT_FCU = "WAIT_FCU"
    STATE_PRESTREAM = "PRESTREAM"
    STATE_SET_MODE = "SET_MODE"
    STATE_ARMING = "ARMING"
    STATE_TAKEOFF = "TAKEOFF"
    STATE_ACTIVE = "ACTIVE"
    STATE_FAILSAFE_HOVER = "FAILSAFE_HOVER"

    MODE_NORMAL = "NORMAL"
    MODE_DEPTH_STOP = "DEPTH_STOP"
    MODE_EMERGENCY_AVOID = "EMERGENCY_AVOID"
    MODE_TARGET_TIMEOUT = "TARGET_TIMEOUT"
    MODE_TARGET_LOST = "TARGET_LOST"
    MODE_RAW_TIMEOUT = "RAW_TIMEOUT"
    MODE_INVALID_CMD = "INVALID_CMD"
    MODE_HEIGHT_GUARD = "HEIGHT_GUARD"

    def __init__(self):
        rospy.init_node("safety_filter_node")

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.target_altitude = float(rospy.get_param("~target_altitude", 1.5))
        self.takeoff_vz = float(rospy.get_param("~takeoff_vz", 0.5))
        self.takeoff_tolerance = float(rospy.get_param("~takeoff_tolerance", 0.1))
        self.prestream_count_target = int(rospy.get_param("~prestream_count", 100))
        self.service_retry_interval = float(rospy.get_param("~service_retry_interval", 1.0))

        self.max_vx = float(rospy.get_param("~max_vx", 0.5))
        self.min_vx = float(rospy.get_param("~min_vx", -0.2))
        self.max_vy = float(rospy.get_param("~max_vy", 0.3))
        self.max_vz = float(rospy.get_param("~max_vz", 0.25))
        self.max_yaw_rate = float(rospy.get_param("~max_yaw_rate", 0.6))

        self.max_delta_vx = float(rospy.get_param("~max_delta_vx", 0.08))
        self.max_delta_vy = float(rospy.get_param("~max_delta_vy", 0.08))
        self.max_delta_vz = float(rospy.get_param("~max_delta_vz", 0.05))
        self.max_delta_yaw_rate = float(rospy.get_param("~max_delta_yaw_rate", 0.08))

        self.min_height = float(rospy.get_param("~min_height", 0.6))
        self.max_height = float(rospy.get_param("~max_height", 2.5))
        self.avoid_stop_depth = float(rospy.get_param("~avoid_stop_depth", 0.6))
        self.avoid_emergency_depth = float(rospy.get_param("~avoid_emergency_depth", 0.35))
        self.danger_area_ratio = float(rospy.get_param("~danger_area_ratio", 0.5))
        self.raw_cmd_timeout = float(rospy.get_param("~raw_cmd_timeout", 0.5))
        self.target_timeout = float(rospy.get_param("~target_timeout", 1.0))
        self.lost_target_slowdown = bool(rospy.get_param("~lost_target_slowdown", True))
        self.set_com_rcl_except = bool(rospy.get_param("~set_com_rcl_except", True))
        self.debug = bool(rospy.get_param("~debug", True))

        self.state = State()
        self.pose = None
        self.velocity = None
        self.raw_cmd = None
        self.raw_stamp = None
        self.target = None
        self.target_stamp = None
        self.risk = None
        self.risk_stamp = None

        self.state_machine_state = self.STATE_WAIT_FCU
        self.safety_mode = self.MODE_NORMAL
        self.prestream_count = 0
        self.last_mode_request = rospy.Time(0)
        self.last_arm_request = rospy.Time(0)
        self.last_filtered_body = BodyCommand()
        self.last_raw_debug = BodyCommand()
        self.last_published_world = BodyCommand()
        self.was_active = False
        self.offboard_params_configured = False

        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_velocity/cmd_vel",
            TwistStamped,
            queue_size=10,
        )
        self.debug_raw_pub = rospy.Publisher(
            "/safety_filter/debug_cmd_raw",
            TwistStamped,
            queue_size=10,
        )
        self.debug_filtered_pub = rospy.Publisher(
            "/safety_filter/debug_cmd_filtered",
            TwistStamped,
            queue_size=10,
        )
        self.mode_pub = rospy.Publisher("/safety_filter/mode", String, queue_size=10)

        rospy.Subscriber("/raw_cmd_vel", TwistStamped, self.raw_cb, queue_size=1)
        rospy.Subscriber("/target/state", TargetState, self.target_cb, queue_size=1)
        rospy.Subscriber("/obstacle/risk", DepthRisk, self.risk_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber(
            "/mavros/local_position/velocity_local",
            TwistStamped,
            self.velocity_cb,
            queue_size=1,
        )
        rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=1)

        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        self.arm_cli = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.mode_cli = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.param_set_cli = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("safety_filter_node ready")

    def raw_cb(self, msg):
        self.raw_cmd = msg
        self.raw_stamp = rospy.Time.now()
        self.last_raw_debug = BodyCommand.from_msg(msg)

    def target_cb(self, msg):
        self.target = msg
        self.target_stamp = rospy.Time.now()

    def risk_cb(self, msg):
        values = [
            msg.front_min_depth,
            msg.front_q05_depth,
            msg.left_q05_depth,
            msg.right_q05_depth,
            msg.obstacle_area_ratio,
        ]
        if not all(finite(value) for value in values):
            rospy.logwarn_throttle(2.0, "Ignoring invalid /obstacle/risk values")
            return
        self.risk = msg
        self.risk_stamp = rospy.Time.now()

    def pose_cb(self, msg):
        self.pose = msg

    def velocity_cb(self, msg):
        self.velocity = msg

    def state_cb(self, msg):
        self.state = msg

    def current_z(self):
        if self.pose is None:
            return 0.0
        return float(self.pose.pose.position.z)

    def current_yaw(self):
        if self.pose is None:
            rospy.logwarn_throttle(2.0, "No /mavros/local_position/pose yet; assuming yaw=0")
            return 0.0
        q = self.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def age(self, stamp):
        if stamp is None:
            return math.inf
        return (rospy.Time.now() - stamp).to_sec()

    def raw_age(self):
        return self.age(self.raw_stamp)

    def target_age(self):
        return self.age(self.target_stamp)

    def service_due(self, last_request):
        return (rospy.Time.now() - last_request).to_sec() >= self.service_retry_interval

    def request_offboard(self):
        if self.state.mode == "OFFBOARD" or not self.service_due(self.last_mode_request):
            return
        self.last_mode_request = rospy.Time.now()
        try:
            response = self.mode_cli(custom_mode="OFFBOARD")
            if not response.mode_sent:
                rospy.logwarn_throttle(2.0, "OFFBOARD request returned mode_sent=False")
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "OFFBOARD request failed: %s", exc)

    def request_arm(self):
        if self.state.armed or not self.service_due(self.last_arm_request):
            return
        self.last_arm_request = rospy.Time.now()
        try:
            response = self.arm_cli(True)
            if not response.success:
                rospy.logwarn_throttle(2.0, "Arming request returned success=False")
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "Arming request failed: %s", exc)

    def configure_offboard_params(self):
        if self.offboard_params_configured or not self.set_com_rcl_except:
            self.offboard_params_configured = True
            return

        try:
            rospy.wait_for_service("/mavros/param/set", timeout=2.0)
            value = ParamValue()
            value.integer = 4
            response = self.param_set_cli("COM_RCL_EXCEPT", value)
            if response.success:
                rospy.loginfo("Set COM_RCL_EXCEPT=4 for no-RC SITL OFFBOARD")
            else:
                rospy.logwarn("COM_RCL_EXCEPT set request returned success=False")
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Failed to set COM_RCL_EXCEPT: %s", exc)
        self.offboard_params_configured = True

    def clamp_body(self, cmd):
        cmd.vx = clamp(cmd.vx, self.min_vx, self.max_vx)
        cmd.vy = clamp(cmd.vy, -self.max_vy, self.max_vy)
        cmd.vz = clamp(cmd.vz, -self.max_vz, self.max_vz)
        cmd.yaw_rate = clamp(cmd.yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        return cmd

    def rate_limit(self, cmd):
        prev = self.last_filtered_body
        cmd.vx = prev.vx + clamp(cmd.vx - prev.vx, -self.max_delta_vx, self.max_delta_vx)
        cmd.vy = prev.vy + clamp(cmd.vy - prev.vy, -self.max_delta_vy, self.max_delta_vy)
        cmd.vz = prev.vz + clamp(cmd.vz - prev.vz, -self.max_delta_vz, self.max_delta_vz)
        cmd.yaw_rate = prev.yaw_rate + clamp(
            cmd.yaw_rate - prev.yaw_rate,
            -self.max_delta_yaw_rate,
            self.max_delta_yaw_rate,
        )
        return cmd

    def apply_target_protection(self, cmd, safety_mode):
        age = self.target_age()
        if age > self.target_timeout:
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.yaw_rate = 0.2
            return cmd, self.MODE_TARGET_TIMEOUT

        if self.target is not None and not self.target.visible and self.lost_target_slowdown:
            cmd.vx = min(cmd.vx, 0.0)
            return cmd, self.MODE_TARGET_LOST

        return cmd, safety_mode

    def apply_obstacle_protection(self, cmd, safety_mode):
        if self.risk is None:
            return cmd, safety_mode

        if self.risk.front_q05_depth < self.avoid_stop_depth:
            cmd.vx = min(cmd.vx, 0.0)
            safety_mode = self.MODE_DEPTH_STOP

        emergency = (
            self.risk.front_q05_depth < self.avoid_emergency_depth
            or (
                self.risk.danger
                and self.risk.obstacle_area_ratio > self.danger_area_ratio
            )
        )
        if emergency:
            cmd.vx = -0.2
            cmd.vy = 0.0
            cmd.vz = max(cmd.vz, 0.1)
            cmd.yaw_rate = 0.3
            safety_mode = self.MODE_EMERGENCY_AVOID

        return cmd, safety_mode

    def apply_height_protection(self, cmd, safety_mode):
        z = self.current_z()
        guarded = False
        if z < self.min_height:
            cmd.vz = max(cmd.vz, 0.1)
            guarded = True
        if z > self.max_height:
            cmd.vz = min(cmd.vz, -0.1)
            guarded = True
        if z < 0.3:
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.vz = max(cmd.vz, 0.1)
            guarded = True
        if guarded and safety_mode == self.MODE_NORMAL:
            safety_mode = self.MODE_HEIGHT_GUARD
        return cmd, safety_mode

    def safe_body_command(self):
        if self.raw_cmd is None or self.raw_age() > self.raw_cmd_timeout:
            cmd, safety_mode = self.apply_height_protection(
                BodyCommand(),
                self.MODE_RAW_TIMEOUT,
            )
            return self.clamp_body(cmd), safety_mode

        raw = BodyCommand.from_msg(self.raw_cmd)
        if not raw.all_finite():
            cmd, safety_mode = self.apply_height_protection(
                BodyCommand(),
                self.MODE_INVALID_CMD,
            )
            return self.clamp_body(cmd), safety_mode

        cmd = self.clamp_body(raw.copy())
        cmd = self.rate_limit(cmd)
        safety_mode = self.MODE_NORMAL
        cmd, safety_mode = self.apply_target_protection(cmd, safety_mode)
        cmd, safety_mode = self.apply_obstacle_protection(cmd, safety_mode)
        cmd, safety_mode = self.apply_height_protection(cmd, safety_mode)
        cmd = self.clamp_body(cmd)
        return cmd, safety_mode

    def body_to_world(self, cmd):
        yaw = self.current_yaw()
        vx_world = math.cos(yaw) * cmd.vx - math.sin(yaw) * cmd.vy
        vy_world = math.sin(yaw) * cmd.vx + math.cos(yaw) * cmd.vy
        return BodyCommand(vx_world, vy_world, cmd.vz, cmd.yaw_rate)

    def make_twist_msg(self, cmd, frame_id):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(cmd.vx)
        msg.twist.linear.y = float(cmd.vy)
        msg.twist.linear.z = float(cmd.vz)
        msg.twist.angular.z = float(cmd.yaw_rate)
        return msg

    def publish_body_debug(self, raw_cmd, filtered_cmd):
        self.debug_raw_pub.publish(self.make_twist_msg(raw_cmd, "base_link"))
        self.debug_filtered_pub.publish(self.make_twist_msg(filtered_cmd, "base_link"))

    def publish_world_velocity(self, cmd):
        msg = self.make_twist_msg(cmd, "map")
        self.setpoint_pub.publish(msg)
        self.last_published_world = cmd.copy()
        return msg

    def publish_zero(self):
        zero = BodyCommand()
        self.safety_mode = self.MODE_NORMAL
        self.last_filtered_body = zero.copy()
        self.publish_body_debug(self.last_raw_debug, zero)
        return self.publish_world_velocity(zero)

    def publish_takeoff(self):
        if self.current_z() < self.target_altitude - self.takeoff_tolerance:
            cmd = BodyCommand(0.0, 0.0, self.takeoff_vz, 0.0)
        else:
            cmd = BodyCommand()
            self.state_machine_state = self.STATE_ACTIVE
            self.was_active = True
            rospy.loginfo("safety_filter_node reached takeoff altitude %.2f m", self.current_z())
        self.safety_mode = self.MODE_NORMAL
        self.last_filtered_body = cmd.copy()
        self.publish_body_debug(self.last_raw_debug, cmd)
        return self.publish_world_velocity(cmd)

    def publish_filtered_active(self):
        body_cmd, self.safety_mode = self.safe_body_command()
        self.last_filtered_body = body_cmd.copy()
        world_cmd = self.body_to_world(body_cmd)
        self.publish_body_debug(self.last_raw_debug, body_cmd)
        return self.publish_world_velocity(world_cmd)

    def publish_mode(self):
        if self.state_machine_state == self.STATE_ACTIVE:
            text = "{}:{}".format(self.STATE_ACTIVE, self.safety_mode)
        elif self.state_machine_state == self.STATE_FAILSAFE_HOVER and self.safety_mode != self.MODE_NORMAL:
            text = "{}:{}".format(self.STATE_FAILSAFE_HOVER, self.safety_mode)
        else:
            text = self.state_machine_state
        self.mode_pub.publish(String(data=text))

    def format_age(self, value):
        if math.isinf(value):
            return -1.0
        return value

    def log_status(self):
        if not self.debug:
            return

        raw = self.last_raw_debug
        filt = self.last_filtered_body
        pub = self.last_published_world
        front_q05 = self.risk.front_q05_depth if self.risk is not None else math.nan
        area_ratio = self.risk.obstacle_area_ratio if self.risk is not None else math.nan
        target_visible = bool(self.target.visible) if self.target is not None else False

        rospy.loginfo_throttle(
            1.0,
            "state_machine_state=%s safety_mode=%s connected=%s armed=%s px4_mode=%s "
            "current_z=%.2f raw_cmd_age=%.2f target_age=%.2f "
            "front_q05_depth=%.2f obstacle_area_ratio=%.2f target_visible=%s "
            "raw[vx=%.2f vy=%.2f vz=%.2f yaw=%.2f] "
            "filtered[vx=%.2f vy=%.2f vz=%.2f yaw=%.2f] "
            "published_world[vx=%.2f vy=%.2f vz=%.2f yaw=%.2f]",
            self.state_machine_state,
            self.safety_mode,
            self.state.connected,
            self.state.armed,
            self.state.mode,
            self.current_z(),
            self.format_age(self.raw_age()),
            self.format_age(self.target_age()),
            front_q05,
            area_ratio,
            target_visible,
            raw.vx,
            raw.vy,
            raw.vz,
            raw.yaw_rate,
            filt.vx,
            filt.vy,
            filt.vz,
            filt.yaw_rate,
            pub.vx,
            pub.vy,
            pub.vz,
            pub.yaw_rate,
        )

    def offboard_or_arm_lost(self):
        return self.was_active and (self.state.mode != "OFFBOARD" or not self.state.armed)

    def recover_offboard_and_arm(self):
        if self.state.mode != "OFFBOARD":
            self.request_offboard()
        if not self.state.armed:
            self.request_arm()

    def step_state_machine(self):
        published_msg = None

        if self.state_machine_state == self.STATE_WAIT_FCU:
            published_msg = self.publish_zero()
            if self.state.connected:
                self.configure_offboard_params()
                rospy.loginfo("MAVROS FCU connected; prestreaming setpoints")
                self.state_machine_state = self.STATE_PRESTREAM
                self.prestream_count = 0

        elif self.state_machine_state == self.STATE_PRESTREAM:
            published_msg = self.publish_zero()
            self.prestream_count += 1
            if self.prestream_count >= self.prestream_count_target:
                rospy.loginfo("Prestreamed %d zero setpoints", self.prestream_count)
                self.state_machine_state = self.STATE_SET_MODE

        elif self.state_machine_state == self.STATE_SET_MODE:
            published_msg = self.publish_zero()
            self.request_offboard()
            if self.state.mode == "OFFBOARD":
                rospy.loginfo("PX4 mode is OFFBOARD; requesting arming")
                self.state_machine_state = self.STATE_ARMING

        elif self.state_machine_state == self.STATE_ARMING:
            published_msg = self.publish_zero()
            if self.state.mode != "OFFBOARD":
                self.state_machine_state = self.STATE_SET_MODE
            else:
                self.request_arm()
                if self.state.armed:
                    rospy.loginfo("PX4 armed; taking off")
                    self.state_machine_state = self.STATE_TAKEOFF

        elif self.state_machine_state == self.STATE_TAKEOFF:
            self.recover_offboard_and_arm()
            published_msg = self.publish_takeoff()

        elif self.state_machine_state == self.STATE_ACTIVE:
            if self.offboard_or_arm_lost():
                self.state_machine_state = self.STATE_FAILSAFE_HOVER
                self.safety_mode = self.MODE_NORMAL
                published_msg = self.publish_zero()
            else:
                published_msg = self.publish_filtered_active()

        elif self.state_machine_state == self.STATE_FAILSAFE_HOVER:
            published_msg = self.publish_zero()
            self.recover_offboard_and_arm()
            if self.state.mode == "OFFBOARD" and self.state.armed:
                if self.current_z() < self.target_altitude - self.takeoff_tolerance:
                    self.state_machine_state = self.STATE_TAKEOFF
                else:
                    self.state_machine_state = self.STATE_ACTIVE

        else:
            rospy.logerr_throttle(1.0, "Unknown safety filter state: %s", self.state_machine_state)
            self.state_machine_state = self.STATE_FAILSAFE_HOVER
            published_msg = self.publish_zero()

        self.publish_mode()
        self.log_status()
        return published_msg

    def run(self):
        while not rospy.is_shutdown():
            self.step_state_machine()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        SafetyFilterNode().run()
    except rospy.ROSInterruptException:
        pass
