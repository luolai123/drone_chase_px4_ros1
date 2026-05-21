#!/usr/bin/env python3

import argparse
import os
import random
import sys

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from std_srvs.srv import Empty


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import spawn_random_obstacles
import spawn_red_ball


def delete_model(delete_proxy, model_name):
    try:
        response = delete_proxy(model_name)
        if response.success:
            rospy.loginfo("Deleted %s", model_name)
    except rospy.ServiceException:
        pass


def delete_chase_models(delete_proxy, world_proxy):
    names = []
    try:
        names = world_proxy().model_names
    except rospy.ServiceException as exc:
        rospy.logwarn("Could not query Gazebo world models: %s", exc)

    delete_model(delete_proxy, spawn_red_ball.MODEL_NAME)
    for name in names:
        if name.startswith(spawn_random_obstacles.OBSTACLE_PREFIX):
            delete_model(delete_proxy, name)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Reset Gazebo world and respawn Phase 2 chase objects.")
    parser.add_argument("--num", type=int, default=4, help="Number of obstacles to spawn.")
    parser.add_argument("--world-size", type=float, default=12.0, help="Square obstacle generation side length in meters.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--red-ball-z", type=float, default=1.0, help="Red ball z position.")
    parser.add_argument("--no-obstacles", action="store_true", help="Respawn only the red ball.")
    parser.add_argument("--model-sdf", default=None, help="Path to red_ball/model.sdf.")
    return parser


def main():
    args = build_arg_parser().parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("reset_chase_world")

    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/get_world_properties")
    rospy.wait_for_service("/gazebo/reset_world")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")

    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
    reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    delete_chase_models(delete_proxy, world_proxy)
    reset_world_proxy()
    rospy.sleep(0.5)

    rng = random.Random(spawn_red_ball.normalize_seed(args.seed))
    ball_x, ball_y, ball_z = spawn_red_ball.choose_position(
        randomize=True,
        x=None,
        y=None,
        z=args.red_ball_z,
        rng=rng,
    )
    model_xml = spawn_red_ball.load_model_xml(args.model_sdf or spawn_red_ball.default_model_sdf_path())
    ball_pose = spawn_red_ball.spawn_red_ball(
        spawn_proxy,
        delete_proxy,
        model_xml,
        ball_x,
        ball_y,
        ball_z,
        replace=False,
    )
    rospy.loginfo(
        "Reset red_ball x=%.3f y=%.3f z=%.3f",
        ball_pose["x"],
        ball_pose["y"],
        ball_pose["z"],
    )

    if not args.no_obstacles:
        obstacle_specs = spawn_random_obstacles.generate_obstacle_specs(
            args.num,
            args.world_size,
            rng,
            red_ball_xy=(ball_pose["x"], ball_pose["y"]),
        )
        obstacles = spawn_random_obstacles.spawn_obstacles(spawn_proxy, obstacle_specs)
        rospy.loginfo("Reset spawned %d obstacles", len(obstacles))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("reset_chase_world failed: %s", exc)
        sys.exit(1)
