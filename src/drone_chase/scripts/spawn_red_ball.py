#!/usr/bin/env python3

import argparse
import os
import random
import sys

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


MODEL_NAME = "red_ball"
POSE_PARAM = "/drone_chase/red_ball_pose"


def find_package_file(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..", relative_path)),
        os.path.abspath(os.path.join(os.getcwd(), "src", "drone_chase", relative_path)),
    ]

    for root in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if root:
            candidates.append(os.path.join(root, "drone_chase", relative_path))

    try:
        import rospkg

        candidates.append(os.path.join(rospkg.RosPack().get_path("drone_chase"), relative_path))
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path

    raise RuntimeError("Cannot find drone_chase/{}".format(relative_path))


def default_model_sdf_path():
    return find_package_file(os.path.join("models", "red_ball", "model.sdf"))


def load_model_xml(path):
    with open(path, "r") as f:
        return f.read()


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def choose_position(randomize=False, x=None, y=None, z=1.0, rng=None):
    rng = rng or random.Random()
    if randomize or x is None or y is None:
        return rng.uniform(3.0, 4.0), rng.uniform(-0.5, 0.5), float(z)
    return float(x), float(y), float(z)


def normalize_seed(seed):
    if seed is None or seed < 0:
        return None
    return seed


def delete_model(delete_proxy, model_name):
    try:
        return delete_proxy(model_name)
    except rospy.ServiceException:
        return None


def model_exists(world_proxy, model_name):
    try:
        return model_name in world_proxy().model_names
    except rospy.ServiceException:
        return False


def spawn_red_ball(spawn_proxy, delete_proxy, model_xml, x, y, z, replace=True, world_proxy=None):
    if replace and (world_proxy is None or model_exists(world_proxy, MODEL_NAME)):
        delete_model(delete_proxy, MODEL_NAME)
        rospy.sleep(0.2)

    response = spawn_proxy(MODEL_NAME, model_xml, "", make_pose(x, y, z), "world")
    if not response.success:
        raise RuntimeError(response.status_message)

    pose_param = {"x": float(x), "y": float(y), "z": float(z)}
    rospy.set_param(POSE_PARAM, pose_param)
    return pose_param


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Spawn the Phase 2 red target ball in Gazebo.")
    parser.add_argument("--x", type=float, default=None, help="Red ball x position in world frame.")
    parser.add_argument("--y", type=float, default=None, help="Red ball y position in world frame.")
    parser.add_argument("--z", type=float, default=1.0, help="Red ball z position in world frame.")
    parser.add_argument("--random", action="store_true", help="Randomize in front of the drone, x=3-5 m.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--model-sdf", default=None, help="Path to red_ball/model.sdf.")
    parser.add_argument("--no-replace", action="store_true", help="Do not delete an existing red_ball first.")
    return parser


def main():
    args = build_arg_parser().parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("spawn_red_ball")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/get_world_properties")
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)

    rng = random.Random(normalize_seed(args.seed))
    x, y, z = choose_position(args.random, args.x, args.y, args.z, rng)
    model_path = args.model_sdf or default_model_sdf_path()
    pose = spawn_red_ball(
        spawn_proxy,
        delete_proxy,
        load_model_xml(model_path),
        x,
        y,
        z,
        replace=not args.no_replace,
        world_proxy=world_proxy,
    )
    rospy.loginfo("Spawned red_ball at x=%.3f y=%.3f z=%.3f", pose["x"], pose["y"], pose["z"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("spawn_red_ball failed: %s", exc)
        sys.exit(1)
