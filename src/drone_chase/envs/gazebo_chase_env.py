#!/usr/bin/env python3
"""Gymnasium-style wrapper around the ROS/Gazebo/PX4 chase loop.

The environment publishes only body-frame semantic commands on /raw_cmd_vel.
The Phase 5 safety_filter_node remains the sole publisher to MAVROS velocity
setpoints.
"""

import math
import os
import random
import subprocess
import threading
import time

import numpy as np
import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from mavros_msgs.msg import State
from std_msgs.msg import String

from drone_chase.msg import DepthRisk, TargetState

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYMNASIUM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    _GYMNASIUM_AVAILABLE = False

try:
    import rospkg
except ImportError:
    rospkg = None


_EnvBase = gym.Env if _GYMNASIUM_AVAILABLE else object


DEFAULT_CONFIG = {
    "step_dt": 0.1,
    "max_episode_steps": 400,
    "max_depth": 10.0,
    "safe_depth": 0.8,
    "capture_distance": 0.55,
    "success_distance": 0.55,
    "success_hold_steps": 3,
    "collision_depth": 0.25,
    "collision_area_ratio": 0.2,
    "approach_weight": 2.0,
    "distance_weight": -0.05,
    "visibility_reward": 0.1,
    "visibility_loss": -0.2,
    "center_weight": 0.2,
    "obstacle_weight": -3.0,
    "obstacle_close_depth": 0.5,
    "obstacle_close_penalty": -1.0,
    "smooth_weight": -0.05,
    "reward_variant": "v1",
    "approach_delta_clip": 100.0,
    "yaw_weight": 0.0,
    "forward_weight": 0.0,
    "lost_extra_weight": 0.0,
    "episode_soft_promote_to_soft_on_oob": True,
    "episode_soft_oob_margin_ratio": 0.9,
    "episode_soft_promote_reasons": ["out_of_bounds", "height_violation"],
    "success_bonus": 50.0,
    "timeout_penalty": -10.0,
    "height_violation_penalty": -30.0,
    "collision_penalty": -50.0,
    "max_vx": 0.5,
    "min_vx": -0.2,
    "max_vy": 0.3,
    "max_vz": 0.25,
    "max_yaw_rate": 0.6,
    "min_episode_height": 0.2,
    "max_episode_height": 3.0,
    "world_x_limit": 8.0,
    "world_y_limit": 8.0,
    "reset_mode": "soft",
    "world_type": "woods_easy",
    "reset_timeout": 25.0,
    "topic_wait_timeout": 15.0,
    "reset_ready_timeout": 15.0,
    "reset_zero_cmd_duration": 1.0,
    "reset_topic_fresh_timeout": 2.0,
    "start_z_min": 1.0,
    "start_z_max": 2.0,
    "disable_done_during_reset": True,
    "initial_grace_steps": 20,
    "respawn_target_on_reset": False,
    "target_spawn_distance_min": 3.0,
    "target_spawn_distance_max": 4.0,
    "target_spawn_lateral_min": -0.5,
    "target_spawn_lateral_max": 0.5,
    "target_spawn_z": 1.0,
    "reset_target_z": 1.2,
    "reset_z_min": 0.85,
    "reset_z_max": 1.65,
    "reset_z_tolerance": 0.15,
    "reset_recovery_timeout": 8.0,
    "reset_recovery_rate": 20.0,
    "reset_max_vz": 0.25,
    "reset_min_visible_wait": 1.0,
    "reset_gate_timeout": 12.0,
    "woods_skip_gazebo_reset_world": True,
    "require_initial_target_visible": True,
    "target_visible_gate_timeout": 5.0,
    "target_visible_min_consecutive": 5,
    "max_target_respawn_attempts": 5,
    "seed": 10,
    "target_x": 4.0,
    "target_y": 0.0,
    "target_z": 1.0,
    "woods_easy_num_trunks": 8,
    "woods_easy_num_branches": 15,
    "woods_easy_num_fallen": 3,
    "woods_easy_area_x_min": 1.5,
    "woods_easy_area_x_max": 6.5,
    "woods_easy_area_y_min": -3.0,
    "woods_easy_area_y_max": 3.0,
    "woods_easy_uav_clearance": 1.5,
    "woods_easy_target_clearance": 0.8,
    "random_woods_num_trunks": 18,
    "random_woods_num_branches": 45,
    "random_woods_num_fallen": 10,
    "random_woods_area_x_min": 1.0,
    "random_woods_area_x_max": 6.5,
    "random_woods_area_y_min": -3.5,
    "random_woods_area_y_max": 3.5,
    "random_woods_uav_clearance": 1.0,
    "random_woods_target_clearance": 0.6,
    "woods_reset_uav_clearance_margin": 0.5,
    "woods_reset_target_relative_to_uav": True,
    "woods_reset_target_distance": 4.0,
    "woods_reset_target_lateral": 0.0,
}


def _package_path():
    if rospkg is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return rospkg.RosPack().get_path("drone_chase")


def _default_config_path():
    return os.path.join(_package_path(), "config", "phase6_env.yaml")


def _load_yaml(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        rospy.logwarn("PyYAML is unavailable; using built-in Phase 6 defaults")
        return {}
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _as_float(config, key):
    return float(config[key])


def _as_int(config, key):
    return int(config[key])


def _as_bool(config, key):
    value = config[key]
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _yaw_from_pose(pose_msg):
    if pose_msg is None:
        return 0.0
    q = pose_msg.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class GazeboChaseEnv(_EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, config_path=None, reset_mode=None, world_type=None, **overrides):
        if not rospy.core.is_initialized():
            rospy.init_node("gazebo_chase_env", anonymous=True, disable_signals=True)

        config = DEFAULT_CONFIG.copy()
        config.update(_load_yaml(config_path or _default_config_path()))
        config.update({key: value for key, value in overrides.items() if value is not None})
        if reset_mode is not None:
            config["reset_mode"] = reset_mode
        if world_type is not None:
            config["world_type"] = world_type
        self.config = config

        self.step_dt = _as_float(config, "step_dt")
        self.max_episode_steps = _as_int(config, "max_episode_steps")
        self.max_depth = _as_float(config, "max_depth")
        self.safe_depth = _as_float(config, "safe_depth")
        self.capture_distance = _as_float(config, "capture_distance")
        self.success_distance = _as_float(config, "success_distance")
        self.success_hold_steps = _as_int(config, "success_hold_steps")
        self.collision_depth = _as_float(config, "collision_depth")
        self.collision_area_ratio = _as_float(config, "collision_area_ratio")
        self.approach_weight = _as_float(config, "approach_weight")
        self.distance_weight = _as_float(config, "distance_weight")
        self.visibility_reward = _as_float(config, "visibility_reward")
        self.visibility_loss = _as_float(config, "visibility_loss")
        self.center_weight = _as_float(config, "center_weight")
        self.obstacle_weight = _as_float(config, "obstacle_weight")
        self.obstacle_close_depth = _as_float(config, "obstacle_close_depth")
        self.obstacle_close_penalty = _as_float(config, "obstacle_close_penalty")
        self.smooth_weight = _as_float(config, "smooth_weight")
        self.reward_variant = str(config.get("reward_variant", "v1"))
        self.approach_delta_clip = _as_float(config, "approach_delta_clip")
        self.yaw_weight = _as_float(config, "yaw_weight")
        self.forward_weight = _as_float(config, "forward_weight")
        self.lost_extra_weight = _as_float(config, "lost_extra_weight")
        self.success_bonus = _as_float(config, "success_bonus")
        self.timeout_penalty = _as_float(config, "timeout_penalty")
        self.height_violation_penalty = _as_float(config, "height_violation_penalty")
        self.collision_penalty = _as_float(config, "collision_penalty")
        self.max_vx = _as_float(config, "max_vx")
        self.min_vx = _as_float(config, "min_vx")
        self.max_vy = _as_float(config, "max_vy")
        self.max_vz = _as_float(config, "max_vz")
        self.max_yaw_rate = _as_float(config, "max_yaw_rate")
        self.min_episode_height = _as_float(config, "min_episode_height")
        self.max_episode_height = _as_float(config, "max_episode_height")
        self.world_x_limit = _as_float(config, "world_x_limit")
        self.world_y_limit = _as_float(config, "world_y_limit")
        self.reset_mode = str(config["reset_mode"])
        self.world_type = str(config["world_type"])
        self.reset_timeout = _as_float(config, "reset_timeout")
        self.topic_wait_timeout = _as_float(config, "topic_wait_timeout")
        self.reset_ready_timeout = _as_float(config, "reset_ready_timeout")
        self.reset_zero_cmd_duration = _as_float(config, "reset_zero_cmd_duration")
        self.reset_topic_fresh_timeout = _as_float(config, "reset_topic_fresh_timeout")
        self.start_z_min = _as_float(config, "start_z_min")
        self.start_z_max = _as_float(config, "start_z_max")
        self.disable_done_during_reset = _as_bool(config, "disable_done_during_reset")
        self.initial_grace_steps = _as_int(config, "initial_grace_steps")
        self.respawn_target_on_reset = _as_bool(config, "respawn_target_on_reset")
        self.reset_target_z = _as_float(config, "reset_target_z")
        self.reset_z_min = _as_float(config, "reset_z_min")
        self.reset_z_max = _as_float(config, "reset_z_max")
        self.reset_z_tolerance = _as_float(config, "reset_z_tolerance")
        self.reset_recovery_timeout = _as_float(config, "reset_recovery_timeout")
        self.reset_recovery_rate = _as_float(config, "reset_recovery_rate")
        self.reset_max_vz = _as_float(config, "reset_max_vz")
        self.reset_min_visible_wait = _as_float(config, "reset_min_visible_wait")
        self.reset_gate_timeout = _as_float(config, "reset_gate_timeout")
        self.woods_skip_gazebo_reset_world = _as_bool(config, "woods_skip_gazebo_reset_world")
        self.require_initial_target_visible = _as_bool(config, "require_initial_target_visible")
        self.target_visible_gate_timeout = _as_float(config, "target_visible_gate_timeout")
        self.target_visible_min_consecutive = _as_int(config, "target_visible_min_consecutive")
        self.max_target_respawn_attempts = _as_int(config, "max_target_respawn_attempts")
        self.woods_reset_uav_clearance_margin = _as_float(config, "woods_reset_uav_clearance_margin")
        self.woods_reset_target_relative_to_uav = _as_bool(config, "woods_reset_target_relative_to_uav")
        self.woods_reset_target_distance = _as_float(config, "woods_reset_target_distance")
        self.woods_reset_target_lateral = _as_float(config, "woods_reset_target_lateral")
        self.last_reset_target_x = _as_float(config, "target_x")
        self.last_reset_target_y = _as_float(config, "target_y")
        self.last_reset_clearance_uav_x = 0.0
        self.last_reset_clearance_uav_y = 0.0
        self.episode_soft_promote_to_soft_on_oob = _as_bool(config, "episode_soft_promote_to_soft_on_oob")
        self.episode_soft_oob_margin_ratio = _as_float(config, "episode_soft_oob_margin_ratio")
        promote_reasons = config.get("episode_soft_promote_reasons", DEFAULT_CONFIG["episode_soft_promote_reasons"])
        if isinstance(promote_reasons, (list, tuple)):
            self.episode_soft_promote_reasons = [str(item) for item in promote_reasons]
        elif promote_reasons is None:
            self.episode_soft_promote_reasons = []
        else:
            self.episode_soft_promote_reasons = [token.strip() for token in str(promote_reasons).split(",") if token.strip()]
        self.target_spawn_distance_min = _as_float(config, "target_spawn_distance_min")
        self.target_spawn_distance_max = _as_float(config, "target_spawn_distance_max")
        self.target_spawn_lateral_min = _as_float(config, "target_spawn_lateral_min")
        self.target_spawn_lateral_max = _as_float(config, "target_spawn_lateral_max")
        self.target_spawn_z = _as_float(config, "target_spawn_z")
        self.rng = random.Random(int(config.get("seed", 10)))

        self.lock = threading.RLock()
        self.target = None
        self.risk = None
        self.pose = None
        self.velocity = None
        self.mavros_state = State()
        self.safety_mode = ""
        self.target_stamp = None
        self.risk_stamp = None
        self.pose_stamp = None
        self.velocity_stamp = None
        self.mode_stamp = None
        self.filtered_cmd = None

        self.episode_steps = 0
        self.success_count = 0
        self.prev_distance = self.max_depth
        self.prev_target_visible = False
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_body_cmd = np.zeros(4, dtype=np.float32)
        self.lost_steps = 0
        self.emergency_count = 0
        self.depth_stop_count = 0
        self.target_lost_count = 0
        self.last_terminal_reason = ""
        self.closed = False

        self.raw_cmd_pub = rospy.Publisher("/raw_cmd_vel", TwistStamped, queue_size=10)
        rospy.Subscriber("/target/state", TargetState, self._target_cb, queue_size=1)
        rospy.Subscriber("/obstacle/risk", DepthRisk, self._risk_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self._velocity_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
        rospy.Subscriber("/safety_filter/mode", String, self._mode_cb, queue_size=1)
        rospy.Subscriber("/mavros/setpoint_velocity/cmd_vel", TwistStamped, self._filtered_cmd_cb, queue_size=1)

        if _GYMNASIUM_AVAILABLE:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            high = np.full((20,), np.finfo(np.float32).max, dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:
            self.action_space = None
            self.observation_space = None

    def _target_cb(self, msg):
        with self.lock:
            self.target = msg
            self.target_stamp = rospy.Time.now()

    def _risk_cb(self, msg):
        with self.lock:
            self.risk = msg
            self.risk_stamp = rospy.Time.now()

    def _pose_cb(self, msg):
        with self.lock:
            self.pose = msg
            self.pose_stamp = rospy.Time.now()

    def _velocity_cb(self, msg):
        with self.lock:
            self.velocity = msg
            self.velocity_stamp = rospy.Time.now()

    def _state_cb(self, msg):
        with self.lock:
            self.mavros_state = msg

    def _mode_cb(self, msg):
        with self.lock:
            self.safety_mode = msg.data
            self.mode_stamp = rospy.Time.now()

    def _filtered_cmd_cb(self, msg):
        with self.lock:
            self.filtered_cmd = msg

    def _snapshot(self):
        with self.lock:
            return {
                "target": self.target,
                "risk": self.risk,
                "pose": self.pose,
                "velocity": self.velocity,
                "mavros_state": self.mavros_state,
                "safety_mode": self.safety_mode,
                "filtered_cmd": self.filtered_cmd,
                "target_stamp": self.target_stamp,
                "risk_stamp": self.risk_stamp,
                "pose_stamp": self.pose_stamp,
                "velocity_stamp": self.velocity_stamp,
                "mode_stamp": self.mode_stamp,
            }

    def _map_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError("action must have shape (4,), got {}".format(action.shape))
        action = np.clip(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        vx = action[0] * self.max_vx if action[0] >= 0.0 else action[0] * abs(self.min_vx)
        return action.astype(np.float32), np.array(
            [
                vx,
                action[1] * self.max_vy,
                action[2] * self.max_vz,
                action[3] * self.max_yaw_rate,
            ],
            dtype=np.float32,
        )

    def _publish_body_cmd(self, body_cmd):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(body_cmd[0])
        msg.twist.linear.y = float(body_cmd[1])
        msg.twist.linear.z = float(body_cmd[2])
        msg.twist.angular.z = float(body_cmd[3])
        self.raw_cmd_pub.publish(msg)

    def _publish_zero_for(self, duration, rate_hz=10.0):
        end_time = time.monotonic() + float(duration)
        zero = np.zeros(4, dtype=np.float32)
        sleep_dt = 1.0 / max(float(rate_hz), 1.0)
        while not rospy.is_shutdown() and time.monotonic() < end_time:
            self._publish_body_cmd(zero)
            time.sleep(sleep_dt)
        self.last_body_cmd = zero.copy()

    def _publish_zero_frames(self, frames=5, rate_hz=10.0):
        zero = np.zeros(4, dtype=np.float32)
        sleep_dt = 1.0 / max(float(rate_hz), 1.0)
        for _ in range(int(frames)):
            if rospy.is_shutdown():
                break
            self._publish_body_cmd(zero)
            time.sleep(sleep_dt)
        self.last_body_cmd = zero.copy()

    def _publish_reset_ready_for(self, duration, rate_hz=10.0):
        end_time = time.monotonic() + float(duration)
        sleep_dt = 1.0 / max(float(rate_hz), 1.0)
        while not rospy.is_shutdown() and time.monotonic() < end_time:
            _, status = self._training_ready_status(require_no_raw_timeout=False)
            self._publish_body_cmd(self._reset_ready_body_cmd(status))
            time.sleep(sleep_dt)
        self.last_body_cmd = np.zeros(4, dtype=np.float32)

    def _target_values(self, target):
        if target is None or not bool(target.visible):
            return False, self.max_depth, 0.0, 0.0, self.max_depth, 0.0, 0.0
        x_body = _finite(target.position_camera.z, self.max_depth)
        y_body = -_finite(target.position_camera.x, 0.0)
        z_body = -_finite(target.position_camera.y, 0.0)
        distance = _finite(target.depth, self.max_depth)
        return True, x_body, y_body, z_body, distance, _finite(target.u, 0.0), _finite(target.v, 0.0)

    def _risk_values(self, risk):
        if risk is None:
            return self.max_depth, self.max_depth, self.max_depth, 0.0, 0.0
        return (
            _finite(risk.front_q05_depth, self.max_depth),
            _finite(risk.left_q05_depth, self.max_depth),
            _finite(risk.right_q05_depth, self.max_depth),
            _finite(risk.obstacle_area_ratio, 0.0),
            1.0 if bool(risk.danger) else 0.0,
        )

    def _body_velocity(self, pose, velocity):
        if velocity is None:
            return 0.0, 0.0, 0.0
        yaw = _yaw_from_pose(pose)
        vx_world = _finite(velocity.twist.linear.x, 0.0)
        vy_world = _finite(velocity.twist.linear.y, 0.0)
        vx_body = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        vy_body = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        return vx_body, vy_body, _finite(velocity.twist.linear.z, 0.0)

    def _filtered_body_cmd(self, pose, filtered_cmd):
        if filtered_cmd is None:
            return float(self.last_body_cmd[0]), float(self.last_body_cmd[1]), float(self.last_body_cmd[2])
        yaw = _yaw_from_pose(pose)
        vx_world = _finite(filtered_cmd.twist.linear.x, 0.0)
        vy_world = _finite(filtered_cmd.twist.linear.y, 0.0)
        vx_body = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        vy_body = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        return vx_body, vy_body, _finite(filtered_cmd.twist.linear.z, 0.0)

    def _get_obs(self):
        snap = self._snapshot()
        visible, tx, ty, tz, distance, u, v = self._target_values(snap["target"])
        front_q05, left_q05, right_q05, area_ratio, obstacle_danger = self._risk_values(snap["risk"])
        vx_body, vy_body, vz = self._body_velocity(snap["pose"], snap["velocity"])
        drone_z = 0.0 if snap["pose"] is None else _finite(snap["pose"].pose.position.z, 0.0)
        obs = np.array(
            [
                1.0 if visible else 0.0,
                tx,
                ty,
                tz,
                distance,
                u,
                v,
                front_q05,
                left_q05,
                right_q05,
                area_ratio,
                obstacle_danger,
                vx_body,
                vy_body,
                vz,
                drone_z,
                self.last_body_cmd[0],
                self.last_body_cmd[1],
                self.last_body_cmd[2],
                self.last_body_cmd[3],
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=self.max_depth, neginf=-self.max_depth).astype(np.float32)

    def _reward_components(self, obs, action, prev_action):
        visible = bool(obs[0] > 0.5)
        current_distance = float(obs[4])
        front_q05 = float(obs[7])
        area_ratio = float(obs[10])
        snap = self._snapshot()
        safety_mode = snap["safety_mode"]
        reward_v2 = self.reward_variant == "v2"

        prev_distance = float(self.prev_distance)
        prev_visible = bool(self.prev_target_visible)
        approach_valid = (
            prev_visible
            and visible
            and prev_distance < self.max_depth
            and current_distance < self.max_depth
        )
        delta_distance = prev_distance - current_distance
        delta_distance_clipped = float(
            np.clip(delta_distance, -abs(self.approach_delta_clip), abs(self.approach_delta_clip))
        )
        if reward_v2:
            r_approach = self.approach_weight * delta_distance_clipped if approach_valid else 0.0
            r_distance = -abs(self.distance_weight) * current_distance
            r_visibility = abs(self.visibility_reward) if visible else -abs(self.visibility_loss)
        else:
            r_approach = self.approach_weight * delta_distance if visible else 0.0
            r_distance = self.distance_weight * current_distance
            r_visibility = self.visibility_reward if visible else self.visibility_loss

        center_error = min(1.0, math.sqrt(float(obs[5]) * float(obs[5]) + float(obs[6]) * float(obs[6])))
        if reward_v2:
            r_center = -abs(self.center_weight) * center_error if visible else 0.0
        else:
            r_center = self.center_weight * (1.0 - center_error) if visible else 0.0

        if front_q05 < self.safe_depth:
            if reward_v2:
                r_obstacle = -abs(self.obstacle_weight) * (self.safe_depth - front_q05)
                if front_q05 < self.obstacle_close_depth:
                    r_obstacle -= abs(self.obstacle_close_penalty)
            else:
                r_obstacle = self.obstacle_weight * (self.safe_depth - front_q05)
                if front_q05 < self.obstacle_close_depth:
                    r_obstacle += self.obstacle_close_penalty
        else:
            r_obstacle = 0.0
        delta = action - prev_action
        if reward_v2:
            r_smooth = -abs(self.smooth_weight) * float(np.dot(delta, delta))
        else:
            r_smooth = self.smooth_weight * float(np.dot(delta, delta))
        r_safety_mode = 0.0
        if "EMERGENCY_AVOID" in safety_mode:
            r_safety_mode = -2.0
        elif "DEPTH_STOP" in safety_mode:
            r_safety_mode = -1.0
        elif "TARGET_LOST" in safety_mode:
            r_safety_mode = -0.5
        elif reward_v2 and "HEIGHT_GUARD" in safety_mode:
            r_safety_mode = -0.3
        elif "RAW_TIMEOUT" in safety_mode:
            r_safety_mode = -1.0

        next_lost_steps = self.lost_steps + 1 if not visible else 0
        r_lost_extra = 0.0
        if reward_v2 and next_lost_steps > 10:
            r_lost_extra = -abs(self.lost_extra_weight) * float(min(next_lost_steps, 50))

        r_yaw = 0.0
        if reward_v2 and visible:
            r_yaw = -abs(self.yaw_weight) * abs(float(action[3]))

        r_forward = 0.0
        if reward_v2 and visible and current_distance > self.success_distance:
            filtered_vx_body, _, _ = self._filtered_body_cmd(snap["pose"], snap["filtered_cmd"])
            unsafe_forward = front_q05 < self.safe_depth or "EMERGENCY" in safety_mode or "DEPTH_STOP" in safety_mode
            if not unsafe_forward:
                r_forward = abs(self.forward_weight) * max(0.0, filtered_vx_body)

        return {
            "r_approach": r_approach,
            "r_distance": r_distance,
            "r_visibility": r_visibility,
            "r_center": r_center,
            "r_obstacle": r_obstacle,
            "r_smooth": r_smooth,
            "r_safety_mode": r_safety_mode,
            "r_lost_extra": r_lost_extra,
            "r_yaw": r_yaw,
            "r_forward": r_forward,
            "r_terminal": 0.0,
            "approach_valid": 1.0 if approach_valid else 0.0,
            "delta_distance": delta_distance,
            "delta_distance_clipped": delta_distance_clipped,
            "prev_distance_used_for_reward": prev_distance,
            "distance_used_for_reward": current_distance,
            "prev_target_visible": 1.0 if prev_visible else 0.0,
            "current_target_visible": 1.0 if visible else 0.0,
            "center_error": center_error,
            "lost_steps": float(next_lost_steps),
            "front_q05_depth": front_q05,
            "obstacle_area_ratio": area_ratio,
        }

    def _done_status(self, obs):
        snap = self._snapshot()
        visible = bool(obs[0] > 0.5)
        distance = float(obs[4])
        front_q05 = float(obs[7])
        area_ratio = float(obs[10])
        drone_z = float(obs[15])
        pose = snap["pose"]
        in_initial_grace = self.disable_done_during_reset and self.episode_steps < self.initial_grace_steps
        collision_condition = front_q05 < self.collision_depth and area_ratio > self.collision_area_ratio
        height_violation_condition = drone_z < self.min_episode_height or drone_z > self.max_episode_height

        if visible and distance < self.success_distance:
            self.success_count += 1
        else:
            self.success_count = 0

        if self.success_count >= self.success_hold_steps:
            return True, False, "success", self.success_bonus
        if collision_condition and not in_initial_grace:
            return True, False, "collision_or_too_close", self.collision_penalty
        if height_violation_condition and not in_initial_grace:
            return True, False, "height_violation", self.height_violation_penalty
        if pose is not None:
            x = _finite(pose.pose.position.x, 0.0)
            y = _finite(pose.pose.position.y, 0.0)
            if abs(x) > self.world_x_limit or abs(y) > self.world_y_limit:
                return True, False, "out_of_bounds", self.height_violation_penalty
        if self.episode_steps >= self.max_episode_steps:
            return False, True, "timeout", self.timeout_penalty
        return False, False, "", 0.0

    def _update_safety_mode_counts(self, safety_mode):
        if "EMERGENCY_AVOID" in safety_mode:
            self.emergency_count += 1
        if "DEPTH_STOP" in safety_mode:
            self.depth_stop_count += 1
        if "TARGET_LOST" in safety_mode:
            self.target_lost_count += 1

    def _info(self, obs, reward_components=None, terminal_reason=""):
        snap = self._snapshot()
        collision_condition = (
            float(obs[7]) < self.collision_depth and float(obs[10]) > self.collision_area_ratio
        )
        height_violation_condition = (
            float(obs[15]) < self.min_episode_height or float(obs[15]) > self.max_episode_height
        )
        pose = snap["pose"]
        drone_x = 0.0 if pose is None else _finite(pose.pose.position.x, 0.0)
        drone_y = 0.0 if pose is None else _finite(pose.pose.position.y, 0.0)
        drone_yaw = _yaw_from_pose(pose)
        return {
            "target_visible": bool(obs[0] > 0.5),
            "target_distance": float(obs[4]),
            "target_u": float(obs[5]),
            "target_v": float(obs[6]),
            "front_q05_depth": float(obs[7]),
            "obstacle_area_ratio": float(obs[10]),
            "obstacle_danger": float(obs[11]),
            "drone_z": float(obs[15]),
            "drone_x": float(drone_x),
            "drone_y": float(drone_y),
            "drone_yaw": float(drone_yaw),
            "safety_mode": snap["safety_mode"],
            "mavros_connected": bool(snap["mavros_state"].connected),
            "mavros_mode": snap["mavros_state"].mode,
            "mavros_armed": bool(snap["mavros_state"].armed),
            "reward_components": reward_components or {},
            "success_distance": self.success_distance,
            "success_count": self.success_count,
            "terminal_reason": terminal_reason,
            "success": terminal_reason == "success",
            "collision": terminal_reason == "collision_or_too_close",
            "height_violation": terminal_reason == "height_violation",
            "collision_condition": bool(collision_condition),
            "height_violation_condition": bool(height_violation_condition),
            "initial_grace": bool(
                self.disable_done_during_reset and self.episode_steps < self.initial_grace_steps
            ),
            "out_of_bounds": terminal_reason == "out_of_bounds",
            "timeout": terminal_reason == "timeout",
            "emergency_count": self.emergency_count,
            "depth_stop_count": self.depth_stop_count,
            "target_lost_count": self.target_lost_count,
            "episode_steps": self.episode_steps,
        }

    def _wait_for_topics(self, timeout=None):
        deadline = time.monotonic() + float(timeout or self.topic_wait_timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            snap = self._snapshot()
            if snap["target"] is not None and snap["risk"] is not None and snap["pose"] is not None:
                return True
            time.sleep(0.05)
        return False

    def _wait_for_active_or_armed(self, timeout=20.0):
        deadline = time.monotonic() + float(timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            snap = self._snapshot()
            if "ACTIVE" in snap["safety_mode"]:
                return True
            if snap["mavros_state"].mode == "OFFBOARD" and snap["mavros_state"].armed:
                return True
            time.sleep(0.1)
        return False

    def _stamp_is_fresh(self, stamp, now, max_age):
        if stamp is None:
            return False
        try:
            return (now - stamp).to_sec() <= max_age
        except Exception:
            return False

    def _training_ready_status(self, require_no_raw_timeout=True):
        snap = self._snapshot()
        state = snap["mavros_state"]
        mode = str(snap["safety_mode"] or "")
        pose = snap["pose"]
        now = rospy.Time.now()
        fresh = all(
            self._stamp_is_fresh(stamp, now, self.reset_topic_fresh_timeout)
            for stamp in (
                snap["target_stamp"],
                snap["risk_stamp"],
                snap["pose_stamp"],
                snap["velocity_stamp"],
                snap["mode_stamp"],
            )
        )
        drone_z = None if pose is None else _finite(pose.pose.position.z, 0.0)
        velocity = snap["velocity"]
        drone_vz = None if velocity is None else _finite(velocity.twist.linear.z, 0.0)
        blocked_safety_modes = ("WAIT_FCU", "PRESTREAM", "SET_MODE", "ARMING", "TAKEOFF")
        status = {
            "connected": bool(getattr(state, "connected", False)),
            "armed": bool(getattr(state, "armed", False)),
            "offboard": str(getattr(state, "mode", "")) == "OFFBOARD",
            "z_ready": drone_z is not None and self.reset_z_min <= drone_z <= self.reset_z_max,
            "vz_ready": drone_vz is not None and abs(drone_vz) <= 0.08,
            "safety_ready": mode != "" and not any(blocked in mode for blocked in blocked_safety_modes),
            "raw_timeout_clear": "RAW_TIMEOUT" not in mode,
            "fresh_topics": bool(fresh),
            "drone_z": drone_z,
            "drone_vz": drone_vz,
            "safety_mode": mode,
            "mavros_mode": str(getattr(state, "mode", "")),
        }
        ready = (
            status["connected"]
            and status["armed"]
            and status["offboard"]
            and status["z_ready"]
            and status["vz_ready"]
            and status["safety_ready"]
            and status["fresh_topics"]
            and (status["raw_timeout_clear"] or not require_no_raw_timeout)
        )
        return bool(ready), status

    def wait_until_training_ready(self, timeout=None):
        deadline = time.monotonic() + float(timeout or self.reset_ready_timeout)
        last_status = {}
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            ready, last_status = self._training_ready_status(require_no_raw_timeout=True)
            if ready:
                return True, last_status
            self._publish_body_cmd(self._reset_ready_body_cmd(last_status))
            time.sleep(0.1)
        return False, last_status

    def _reset_ready_body_cmd(self, status):
        cmd = np.zeros(4, dtype=np.float32)
        drone_z = status.get("drone_z")
        if drone_z is None:
            return cmd
        drone_vz = _finite(status.get("drone_vz"), 0.0)
        target_z = self.reset_target_z
        correction_vz = max(0.05, min(abs(self.reset_max_vz), abs(self.max_vz)))
        if drone_z < target_z - abs(self.reset_z_tolerance):
            cmd[2] = correction_vz
        elif drone_z > target_z + abs(self.reset_z_tolerance):
            cmd[2] = -correction_vz
        elif drone_vz > 0.08:
            cmd[2] = -min(0.1, correction_vz)
        elif drone_vz < -0.08:
            cmd[2] = min(0.1, correction_vz)
        return cmd

    def _recover_altitude_for_reset(self):
        start_time = time.monotonic()
        deadline = start_time + max(0.1, float(self.reset_recovery_timeout))
        rate_hz = max(1.0, float(self.reset_recovery_rate))
        sleep_dt = 1.0 / rate_hz
        max_vz = abs(float(self.reset_max_vz))
        used = False
        last_z = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            snap = self._snapshot()
            pose = snap.get("pose")
            if pose is None:
                self._publish_body_cmd(np.zeros(4, dtype=np.float32))
                time.sleep(sleep_dt)
                continue
            current_z = _finite(pose.pose.position.z, 0.0)
            last_z = current_z
            if self.reset_z_min <= current_z <= self.reset_z_max:
                self._publish_zero_for(0.5, rate_hz=rate_hz)
                return True, used, time.monotonic() - start_time, current_z
            z_error = float(self.reset_target_z) - current_z
            vz = float(np.clip(0.6 * z_error, -max_vz, max_vz))
            if current_z < 0.6:
                vz = max(vz, min(max_vz, 0.15))
            cmd = np.zeros(4, dtype=np.float32)
            cmd[2] = vz
            used = True
            self._publish_body_cmd(cmd)
            time.sleep(sleep_dt)
        return False, used, time.monotonic() - start_time, last_z

    def _wait_for_target_visible_gate(self, timeout=None, min_consecutive=None):
        timeout = float(timeout if timeout is not None else self.target_visible_gate_timeout)
        min_consecutive = int(min_consecutive if min_consecutive is not None else self.target_visible_min_consecutive)
        deadline = time.monotonic() + max(0.1, timeout)
        consecutive = 0
        total_visible = 0
        total_samples = 0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            snap = self._snapshot()
            target = snap.get("target")
            visible = bool(target is not None and target.visible)
            total_samples += 1
            total_visible += int(visible)
            consecutive = consecutive + 1 if visible else 0
            self._publish_body_cmd(np.zeros(4, dtype=np.float32))
            if consecutive >= max(1, min_consecutive):
                return True, {
                    "consecutive": int(consecutive),
                    "samples": int(total_samples),
                    "visible_samples": int(total_visible),
                }
            time.sleep(0.05)
        return False, {
            "consecutive": int(consecutive),
            "samples": int(total_samples),
            "visible_samples": int(total_visible),
        }

    def _red_ball_model_xml(self):
        model_path = os.path.join(_package_path(), "models", "red_ball", "model.sdf")
        with open(model_path, "r") as handle:
            return handle.read()

    def _make_spawn_pose(self, x, y, z):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0
        return pose

    def _respawn_target_relative_to_drone(self):
        if self.world_type not in ("world_0", "world_1"):
            return True, "skipped_for_world_type"
        snap = self._snapshot()
        pose_msg = snap["pose"]
        if pose_msg is None:
            return False, "missing_drone_pose"

        yaw = _yaw_from_pose(pose_msg)
        distance = self.rng.uniform(self.target_spawn_distance_min, self.target_spawn_distance_max)
        lateral = self.rng.uniform(self.target_spawn_lateral_min, self.target_spawn_lateral_max)
        x_uav = _finite(pose_msg.pose.position.x, 0.0)
        y_uav = _finite(pose_msg.pose.position.y, 0.0)
        x_target = x_uav + distance * math.cos(yaw) - lateral * math.sin(yaw)
        y_target = y_uav + distance * math.sin(yaw) + lateral * math.cos(yaw)
        z_target = self.target_spawn_z

        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=2.0)
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=2.0)
            delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            try:
                delete_proxy("red_ball")
                rospy.sleep(0.2)
            except rospy.ServiceException:
                pass
            response = spawn_proxy(
                "red_ball",
                self._red_ball_model_xml(),
                "",
                self._make_spawn_pose(x_target, y_target, z_target),
                "world",
            )
            if not response.success:
                return False, str(response.status_message)
            rospy.set_param(
                "/drone_chase/red_ball_pose",
                {"x": float(x_target), "y": float(y_target), "z": float(z_target)},
            )
            return True, "x={:.3f} y={:.3f} z={:.3f}".format(x_target, y_target, z_target)
        except (OSError, rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Episode target respawn failed: %s", exc)
            return False, str(exc)

    def _run_reset_script(self, seed_override=None):
        package_path = _package_path()
        scripts_dir = os.path.join(package_path, "scripts")
        snap = self._snapshot()
        pose = snap.get("pose")
        uav_x = _finite(pose.pose.position.x, 0.0) if pose is not None else 0.0
        uav_y = _finite(pose.pose.position.y, 0.0) if pose is not None else 0.0
        target_x = float(self.config.get("target_x", 4.0))
        target_y = float(self.config.get("target_y", 0.0))
        target_z = float(self.config.get("target_z", 1.0))
        if (
            self.world_type in ("woods_easy", "woods_hard", "random_woods", "woods")
            and self.woods_reset_target_relative_to_uav
            and pose is not None
        ):
            yaw = _yaw_from_pose(pose)
            distance = max(0.1, float(self.woods_reset_target_distance))
            lateral = float(self.woods_reset_target_lateral)
            target_x = uav_x + distance * math.cos(yaw) - lateral * math.sin(yaw)
            target_y = uav_y + distance * math.sin(yaw) + lateral * math.cos(yaw)
        self.last_reset_target_x = float(target_x)
        self.last_reset_target_y = float(target_y)
        self.last_reset_clearance_uav_x = float(uav_x)
        self.last_reset_clearance_uav_y = float(uav_y)
        if self.world_type in ("woods_easy", "woods_hard", "random_woods", "woods"):
            cmd = [
                os.path.join(scripts_dir, "reset_random_woods_world.py"),
                "--seed",
                str(int(seed_override if seed_override is not None else self.config.get("seed", 10))),
                "--target-x",
                str(float(target_x)),
                "--target-y",
                str(float(target_y)),
                "--target-z",
                str(float(target_z)),
                "--uav-x",
                str(float(uav_x)),
                "--uav-y",
                str(float(uav_y)),
            ]
            if self.woods_skip_gazebo_reset_world:
                cmd.append("--skip-reset-world")
            woods_param_prefix = None
            if self.world_type == "woods_easy":
                woods_param_prefix = "woods_easy"
            elif self.world_type == "random_woods":
                woods_param_prefix = "random_woods"
            if woods_param_prefix is not None:
                cmd.extend(
                    [
                        "--num-trunks",
                        str(int(self.config["{}_num_trunks".format(woods_param_prefix)])),
                        "--num-branches",
                        str(int(self.config["{}_num_branches".format(woods_param_prefix)])),
                        "--num-fallen",
                        str(int(self.config["{}_num_fallen".format(woods_param_prefix)])),
                        "--area-x-min",
                        str(float(self.config["{}_area_x_min".format(woods_param_prefix)])),
                        "--area-x-max",
                        str(float(self.config["{}_area_x_max".format(woods_param_prefix)])),
                        "--area-y-min",
                        str(float(self.config["{}_area_y_min".format(woods_param_prefix)])),
                        "--area-y-max",
                        str(float(self.config["{}_area_y_max".format(woods_param_prefix)])),
                        "--uav-clearance",
                        str(
                            float(self.config["{}_uav_clearance".format(woods_param_prefix)])
                            + max(0.0, float(self.woods_reset_uav_clearance_margin))
                        ),
                        "--target-clearance",
                        str(float(self.config["{}_target_clearance".format(woods_param_prefix)])),
                    ]
                )
        else:
            cmd = [os.path.join(scripts_dir, "reset_chase_world.py")]
            if self.world_type == "world_0":
                cmd.append("--no-obstacles")
            if "seed" in self.config:
                cmd.extend(["--seed", str(int(self.config["seed"]))])
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            deadline = time.monotonic() + float(self.reset_timeout)
            rate_hz = max(1.0, float(self.reset_recovery_rate))
            sleep_dt = 1.0 / rate_hz
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    proc.kill()
                    stdout, _stderr = proc.communicate()
                    output = (stdout or "") + "\nreset script timeout after {:.1f}s".format(float(self.reset_timeout))
                    rospy.logwarn("Reset script timed out output=%s", output[-1000:])
                    return False, output
                ready, status = self._training_ready_status(require_no_raw_timeout=False)
                self._publish_body_cmd(self._reset_ready_body_cmd(status))
                time.sleep(sleep_dt)
            stdout, _stderr = proc.communicate()
            output = stdout or ""
            if proc.returncode != 0:
                rospy.logwarn("Reset script failed rc=%s output=%s", proc.returncode, output[-1000:])
                return False, output
            return True, output
        except OSError as exc:
            rospy.logwarn("Reset script execution failed: %s", exc)
            return False, str(exc)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(int(seed))
        options = options or {}
        requested_reset_mode = str(options.get("reset_mode", self.reset_mode))
        reset_mode = requested_reset_mode
        promote_to_soft = False
        promote_reason = ""

        if requested_reset_mode == "episode_soft" and self.episode_soft_promote_to_soft_on_oob:
            if self.last_terminal_reason in set(self.episode_soft_promote_reasons):
                promote_to_soft = True
                promote_reason = "last_terminal_reason={}".format(self.last_terminal_reason)
            else:
                snap = self._snapshot()
                pose = snap.get("pose")
                if pose is not None:
                    x = abs(_finite(pose.pose.position.x, 0.0))
                    y = abs(_finite(pose.pose.position.y, 0.0))
                    margin = max(0.0, min(0.999, abs(self.episode_soft_oob_margin_ratio)))
                    if (
                        self.world_x_limit > 0.0
                        and self.world_y_limit > 0.0
                        and (x > self.world_x_limit * margin or y > self.world_y_limit * margin)
                    ):
                        promote_to_soft = True
                        promote_reason = "pose_near_oob x={:.2f} y={:.2f} margin={:.3f}".format(x, y, margin)

        if promote_to_soft:
            reset_mode = "soft"
        self.episode_steps = 0
        self.success_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_body_cmd = np.zeros(4, dtype=np.float32)
        self.prev_target_visible = False
        self.lost_steps = 0
        self.emergency_count = 0
        self.depth_stop_count = 0
        self.target_lost_count = 0
        reset_success = True
        reset_output = ""
        topics_ready = False
        training_ready = False
        training_ready_status = {}
        target_respawn_success = True
        target_respawn_message = ""
        reset_started_at = time.monotonic()
        is_woods_reset = self.world_type in ("woods_easy", "woods_hard", "random_woods", "woods")
        altitude_recovery_used = False
        altitude_recovery_success = False
        altitude_recovery_duration = 0.0
        altitude_recovery_last_z = None
        target_visible_gate_success = not (is_woods_reset and self.require_initial_target_visible)
        target_visible_gate_status = {}
        target_visible_initial_before_gate = False
        target_respawn_attempts = 0

        zero_duration = self.reset_zero_cmd_duration if reset_mode in ("none", "soft", "episode_soft") else 0.2
        if reset_mode == "episode_soft" or (reset_mode == "soft" and is_woods_reset):
            self._publish_reset_ready_for(zero_duration)
        else:
            self._publish_zero_for(zero_duration)
        if reset_mode == "soft":
            if is_woods_reset:
                max_attempts = max(1, int(self.max_target_respawn_attempts if self.require_initial_target_visible else 1))
                base_seed = int(self.config.get("seed", 10))
                reset_success = False
                self._wait_for_topics(timeout=1.0)
                pre_altitude_ok, pre_altitude_used, pre_altitude_duration, pre_last_z = self._recover_altitude_for_reset()
                altitude_recovery_used = bool(altitude_recovery_used or pre_altitude_used)
                altitude_recovery_success = bool(pre_altitude_ok)
                altitude_recovery_duration += float(pre_altitude_duration)
                altitude_recovery_last_z = pre_last_z
                for attempt in range(max_attempts):
                    target_respawn_attempts = attempt + 1
                    attempt_seed = base_seed + attempt * 100000
                    reset_success, reset_output = self._run_reset_script(seed_override=attempt_seed)
                    self._wait_for_topics(timeout=1.0)
                    altitude_ok, altitude_used, altitude_duration, last_z = self._recover_altitude_for_reset()
                    altitude_recovery_used = bool(altitude_recovery_used or altitude_used)
                    altitude_recovery_success = bool(altitude_recovery_success and altitude_ok)
                    altitude_recovery_duration += float(altitude_duration)
                    altitude_recovery_last_z = last_z
                    training_ready, training_ready_status = self.wait_until_training_ready(self.reset_gate_timeout)
                    topics_ready = bool(training_ready_status.get("fresh_topics", False))
                    snap_before_gate = self._snapshot()
                    target_before_gate = snap_before_gate.get("target")
                    if attempt == 0:
                        target_visible_initial_before_gate = bool(target_before_gate is not None and target_before_gate.visible)
                    if self.require_initial_target_visible:
                        target_visible_gate_success, target_visible_gate_status = self._wait_for_target_visible_gate(
                            timeout=self.target_visible_gate_timeout,
                            min_consecutive=self.target_visible_min_consecutive,
                        )
                    else:
                        target_visible_gate_success = True
                        target_visible_gate_status = {"consecutive": 0, "samples": 0, "visible_samples": 0}
                    final_altitude_ok, final_altitude_used, final_altitude_duration, final_last_z = self._recover_altitude_for_reset()
                    altitude_recovery_used = bool(altitude_recovery_used or final_altitude_used)
                    altitude_recovery_success = bool(altitude_recovery_success and final_altitude_ok)
                    altitude_recovery_duration += float(final_altitude_duration)
                    altitude_recovery_last_z = final_last_z
                    self._publish_reset_ready_for(max(0.1, float(self.reset_min_visible_wait)))
                    training_ready, training_ready_status = self._training_ready_status(require_no_raw_timeout=True)
                    topics_ready = bool(training_ready_status.get("fresh_topics", False))
                    if reset_success and altitude_ok and final_altitude_ok and training_ready and topics_ready and target_visible_gate_success:
                        break
                target_respawn_success = bool(target_visible_gate_success)
                target_respawn_message = "visible_gate={} attempts={} status={}".format(
                    bool(target_visible_gate_success),
                    target_respawn_attempts,
                    target_visible_gate_status,
                )
                reset_success = bool(reset_success and altitude_recovery_success and training_ready and target_respawn_success)
            else:
                reset_success, reset_output = self._run_reset_script()
                self._wait_for_topics(timeout=1.0)
                training_ready, training_ready_status = self.wait_until_training_ready(self.reset_ready_timeout)
                if training_ready:
                    if self.respawn_target_on_reset:
                        target_respawn_success, target_respawn_message = self._respawn_target_relative_to_drone()
                        self._wait_for_topics(timeout=1.0)
                    self._publish_zero_for(self.reset_zero_cmd_duration)
                    self._publish_zero_frames(5)
                    training_ready, training_ready_status = self._training_ready_status(require_no_raw_timeout=True)
                topics_ready = bool(training_ready_status.get("fresh_topics", False))
                reset_success = bool(reset_success and training_ready and target_respawn_success)
        elif reset_mode == "hard":
            rospy.logwarn("hard reset is experimental and not fully implemented.")
            reset_success = False
        elif reset_mode == "episode_soft":
            self._wait_for_topics(timeout=1.0)
            if self.respawn_target_on_reset:
                target_respawn_success, target_respawn_message = self._respawn_target_relative_to_drone()
            training_ready, training_ready_status = self.wait_until_training_ready(self.reset_ready_timeout)
            if training_ready:
                self._publish_zero_for(self.reset_zero_cmd_duration)
                self._publish_zero_frames(5)
                training_ready, training_ready_status = self._training_ready_status(require_no_raw_timeout=True)
            topics_ready = bool(training_ready_status.get("fresh_topics", False))
            reset_success = bool(reset_success and training_ready and target_respawn_success)
        elif reset_mode != "none":
            rospy.logwarn("Unknown reset_mode=%s; using no-reset behavior", reset_mode)
            reset_success = False

        if reset_mode not in ("episode_soft", "soft"):
            topics_ready = self._wait_for_topics()

        obs = self._get_obs()
        self.prev_distance = float(obs[4])
        self.prev_target_visible = bool(obs[0] > 0.5)
        info = self._info(obs)
        info["reset_success"] = bool(reset_success and topics_ready)
        info["topics_ready"] = bool(topics_ready)
        info["training_ready"] = bool(training_ready)
        info["training_ready_status"] = training_ready_status
        info["target_respawn_success"] = bool(target_respawn_success)
        info["target_respawn_message"] = target_respawn_message
        info["reset_mode"] = reset_mode
        info["reset_mode_requested"] = requested_reset_mode
        info["reset_mode_used"] = reset_mode
        info["reset_promoted_to_soft"] = bool(promote_to_soft)
        info["reset_promote_reason"] = promote_reason
        info["reset_output_tail"] = reset_output[-1000:] if reset_output else ""
        info["altitude_recovery_used"] = bool(altitude_recovery_used)
        info["altitude_recovery_success"] = bool(altitude_recovery_success)
        info["altitude_recovery_duration"] = float(altitude_recovery_duration)
        info["altitude_recovery_last_z"] = _finite(altitude_recovery_last_z, float(obs[15]))
        info["target_visible_gate_success"] = bool(target_visible_gate_success)
        info["target_visible_gate_status"] = target_visible_gate_status
        info["target_visible_initial_before_gate"] = bool(target_visible_initial_before_gate)
        info["target_respawn_attempts"] = int(target_respawn_attempts)
        info["reset_target_x"] = float(self.last_reset_target_x)
        info["reset_target_y"] = float(self.last_reset_target_y)
        info["reset_clearance_uav_x"] = float(self.last_reset_clearance_uav_x)
        info["reset_clearance_uav_y"] = float(self.last_reset_clearance_uav_y)
        info["reset_duration"] = float(time.monotonic() - reset_started_at)
        return obs, info

    def step(self, action):
        previous_action = self.prev_action.copy()
        action, body_cmd = self._map_action(action)
        self._publish_body_cmd(body_cmd)
        self.last_body_cmd = body_cmd.copy()
        try:
            rospy.sleep(self.step_dt)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                time.sleep(self.step_dt)

        self.episode_steps += 1
        obs = self._get_obs()
        components = self._reward_components(obs, action, previous_action)
        terminated, truncated, terminal_reason, terminal_reward = self._done_status(obs)
        self.last_terminal_reason = str(terminal_reason or "")
        self._update_safety_mode_counts(self._snapshot()["safety_mode"])
        components["r_terminal"] = terminal_reward
        reward = float(sum(components[key] for key in (
            "r_approach",
            "r_distance",
            "r_visibility",
            "r_center",
            "r_obstacle",
            "r_smooth",
            "r_safety_mode",
            "r_lost_extra",
            "r_yaw",
            "r_forward",
            "r_terminal",
        )))
        reward = _finite(reward, 0.0)
        self.prev_action = action.copy()
        self.prev_distance = float(obs[4])
        self.prev_target_visible = bool(obs[0] > 0.5)
        self.lost_steps = int(float(components.get("lost_steps", 0.0)))
        info = self._info(obs, components, terminal_reason)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.closed:
            return
        self._publish_body_cmd(np.zeros(4, dtype=np.float32))
        self.closed = True
