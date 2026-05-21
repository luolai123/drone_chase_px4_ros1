#!/usr/bin/env python3
"""Spawn or delete a diagnostic front obstacle for risk checks."""

import argparse
import math
import sys

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, PoseStamped


MODEL_NAME = "diagnostic_front_obstacle"


def option_present(argv, option_name):
    return any(arg == option_name or arg.startswith(option_name + "=") for arg in argv)


def obstacle_sdf(name, x_size, y_size, z_size):
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{x_size:.3f} {y_size:.3f} {z_size:.3f}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{x_size:.3f} {y_size:.3f} {z_size:.3f}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.1 0.1 0.9 1</ambient>
          <diffuse>0.1 0.1 0.9 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(name=name, x_size=x_size, y_size=y_size, z_size=z_size)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def make_pose(x, y, z, yaw=0.0):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def delete_if_exists(delete_proxy):
    try:
        response = delete_proxy(MODEL_NAME)
        rospy.sleep(0.2)
        return bool(response.success)
    except rospy.ServiceException:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Spawn a diagnostic box obstacle in front of the Iris.")
    parser.add_argument("--x", type=float, default=1.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.75)
    parser.add_argument("--x-size", type=float, default=0.30)
    parser.add_argument("--y-size", type=float, default=1.20)
    parser.add_argument("--z-size", type=float, default=2.00)
    parser.add_argument("--delete-only", action="store_true")
    parser.add_argument("--relative-to-uav", action="store_true")
    parser.add_argument("--distance", type=float, default=0.65)
    parser.add_argument("--front-face-distance", type=float, default=0.45)
    parser.add_argument("--center-z-mode", choices=("uav", "fixed"), default="uav")
    parser.add_argument("--yaw-align", dest="yaw_align", action="store_true", default=True)
    parser.add_argument("--no-yaw-align", dest="yaw_align", action="store_false")
    parser.add_argument("--wait-pose-timeout", type=float, default=5.0)
    cli_args = rospy.myargv(argv=sys.argv)[1:]
    args = parser.parse_args(cli_args)
    distance_requested = option_present(cli_args, "--distance")
    front_face_requested = option_present(cli_args, "--front-face-distance")
    args.use_front_face_distance = front_face_requested or not distance_requested
    validate_args(args)
    return args


def validate_args(args):
    if args.x_size <= 0.0 or args.y_size <= 0.0 or args.z_size <= 0.0:
        raise ValueError("obstacle dimensions must be positive")
    if args.distance <= 0.0:
        raise ValueError("--distance must be positive")
    if args.front_face_distance < 0.0:
        raise ValueError("--front-face-distance must be non-negative")
    if args.wait_pose_timeout <= 0.0:
        raise ValueError("--wait-pose-timeout must be positive")


def wait_for_uav_pose(timeout):
    rospy.loginfo("Waiting for /mavros/local_position/pose for up to %.1f s", timeout)
    return rospy.wait_for_message("/mavros/local_position/pose", PoseStamped, timeout=timeout)


def compute_obstacle_pose(args):
    if not args.relative_to_uav:
        return make_pose(args.x, args.y, args.z), None, 0.0, None, None

    pose_msg = wait_for_uav_pose(args.wait_pose_timeout)
    uav_position = pose_msg.pose.position
    uav_yaw = yaw_from_quaternion(pose_msg.pose.orientation)

    if args.use_front_face_distance:
        center_distance = args.front_face_distance + 0.5 * args.x_size
        expected_front_face_distance = args.front_face_distance
        distance_mode = "front_face"
    else:
        center_distance = args.distance
        expected_front_face_distance = max(0.0, args.distance - 0.5 * args.x_size)
        distance_mode = "center"

    x_obs = uav_position.x + center_distance * math.cos(uav_yaw)
    y_obs = uav_position.y + center_distance * math.sin(uav_yaw)
    if args.center_z_mode == "uav":
        z_obs = uav_position.z
    else:
        z_obs = args.z

    obstacle_yaw = uav_yaw if args.yaw_align else 0.0
    obstacle_pose = make_pose(x_obs, y_obs, z_obs, obstacle_yaw)
    rospy.loginfo(
        "UAV pose: x=%.3f y=%.3f z=%.3f",
        uav_position.x,
        uav_position.y,
        uav_position.z,
    )
    rospy.loginfo(
        "UAV yaw: %.3f rad",
        uav_yaw,
    )
    rospy.loginfo(
        "Relative placement: distance_mode=%s center_distance=%.3f expected_front_face_distance=%.3f",
        distance_mode,
        center_distance,
        expected_front_face_distance,
    )
    return obstacle_pose, pose_msg, uav_yaw, center_distance, expected_front_face_distance


def main():
    args = parse_args()
    rospy.init_node("diagnostic_spawn_front_obstacle")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    deleted_old = delete_if_exists(delete_proxy)
    if args.delete_only:
        rospy.loginfo("Deleted %s: %s", MODEL_NAME, deleted_old)
        return

    obstacle_pose, _uav_pose, uav_yaw, center_distance, front_face_distance = compute_obstacle_pose(args)
    response = spawn_proxy(
        MODEL_NAME,
        obstacle_sdf(MODEL_NAME, args.x_size, args.y_size, args.z_size),
        "",
        obstacle_pose,
        "world",
    )
    if not response.success:
        raise RuntimeError(response.status_message)

    obstacle_yaw = yaw_from_quaternion(obstacle_pose.orientation)
    rospy.loginfo(
        "Obstacle pose: name=%s x=%.3f y=%.3f z=%.3f yaw=%.3f mode=%s",
        MODEL_NAME,
        obstacle_pose.position.x,
        obstacle_pose.position.y,
        obstacle_pose.position.z,
        obstacle_yaw,
        "relative_to_uav" if args.relative_to_uav else "world",
    )
    rospy.loginfo(
        "Obstacle size: x=%.3f y=%.3f z=%.3f",
        args.x_size,
        args.y_size,
        args.z_size,
    )
    if args.relative_to_uav:
        rospy.loginfo(
            "Expected front face distance: %.3f m (center_distance=%.3f m, yaw_align=%s, uav_yaw=%.3f rad)",
            front_face_distance,
            center_distance,
            args.yaw_align,
            uav_yaw,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("diagnostic_spawn_front_obstacle failed: %s", exc)
        sys.exit(1)
