#!/usr/bin/env python3

import argparse
import os
import sys

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from std_srvs.srv import Empty


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import spawn_random_woods
import spawn_red_ball


def delete_model(delete_proxy, model_name):
    try:
        response = delete_proxy(model_name)
        if response.success:
            rospy.loginfo("Deleted %s", model_name)
        return response
    except rospy.ServiceException:
        return None


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Reset Gazebo and respawn red_ball plus random_woods.")
    parser.add_argument("--seed", type=int, default=spawn_random_woods.DEFAULTS["seed"], help="Random woods seed.")
    parser.add_argument("--target-x", type=float, default=spawn_random_woods.DEFAULTS["target_x"], help="Red ball x position.")
    parser.add_argument("--target-y", type=float, default=spawn_random_woods.DEFAULTS["target_y"], help="Red ball y position.")
    parser.add_argument("--target-z", type=float, default=1.0, help="Red ball z position.")
    parser.add_argument("--num-trunks", type=int, default=spawn_random_woods.DEFAULTS["num_trunks"], help="Number of trunks.")
    parser.add_argument("--num-branches", type=int, default=spawn_random_woods.DEFAULTS["num_branches"], help="Number of branches.")
    parser.add_argument("--num-fallen", type=int, default=spawn_random_woods.DEFAULTS["num_fallen"], help="Number of fallen logs.")
    parser.add_argument("--uav-x", type=float, default=spawn_random_woods.DEFAULTS["uav_x"], help="UAV x used for clearance.")
    parser.add_argument("--uav-y", type=float, default=spawn_random_woods.DEFAULTS["uav_y"], help="UAV y used for clearance.")
    parser.add_argument("--area-x-min", type=float, default=spawn_random_woods.DEFAULTS["area_x_min"], help="Minimum obstacle center x.")
    parser.add_argument("--area-x-max", type=float, default=spawn_random_woods.DEFAULTS["area_x_max"], help="Maximum obstacle center x.")
    parser.add_argument("--area-y-min", type=float, default=spawn_random_woods.DEFAULTS["area_y_min"], help="Minimum obstacle center y.")
    parser.add_argument("--area-y-max", type=float, default=spawn_random_woods.DEFAULTS["area_y_max"], help="Maximum obstacle center y.")
    parser.add_argument("--uav-clearance", type=float, default=spawn_random_woods.DEFAULTS["uav_clearance"], help="XY clearance around the UAV start.")
    parser.add_argument("--target-clearance", type=float, default=spawn_random_woods.DEFAULTS["target_clearance"], help="XY clearance around red_ball.")
    parser.add_argument("--max-attempts", type=int, default=spawn_random_woods.DEFAULTS["max_attempts"], help="Placement attempts per woods element.")
    parser.add_argument("--model-sdf", default=None, help="Path to red_ball/model.sdf.")
    parser.add_argument(
        "--skip-reset-world",
        action="store_true",
        help="Do not call /gazebo/reset_world; only respawn red_ball and random_woods.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("reset_random_woods_world")

    rospy.wait_for_service("/gazebo/delete_model")
    if not args.skip_reset_world:
        rospy.wait_for_service("/gazebo/reset_world")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")

    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty) if not args.skip_reset_world else None
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    delete_model(delete_proxy, spawn_random_woods.MODEL_NAME)
    delete_model(delete_proxy, spawn_red_ball.MODEL_NAME)
    if reset_world_proxy is not None:
        reset_world_proxy()
        rospy.sleep(0.5)
    else:
        rospy.sleep(0.2)

    model_xml = spawn_red_ball.load_model_xml(args.model_sdf or spawn_red_ball.default_model_sdf_path())
    ball_pose = spawn_red_ball.spawn_red_ball(
        spawn_proxy,
        delete_proxy,
        model_xml,
        args.target_x,
        args.target_y,
        args.target_z,
        replace=False,
    )
    rospy.loginfo(
        "Respawned red_ball x=%.3f y=%.3f z=%.3f",
        ball_pose["x"],
        ball_pose["y"],
        ball_pose["z"],
    )

    specs = spawn_random_woods.generate_woods_specs(
        seed=args.seed,
        area_x_min=args.area_x_min,
        area_x_max=args.area_x_max,
        area_y_min=args.area_y_min,
        area_y_max=args.area_y_max,
        num_trunks=args.num_trunks,
        num_branches=args.num_branches,
        num_fallen=args.num_fallen,
        uav_x=args.uav_x,
        uav_y=args.uav_y,
        uav_clearance=args.uav_clearance,
        target_x=args.target_x,
        target_y=args.target_y,
        target_clearance=args.target_clearance,
        max_attempts=args.max_attempts,
    )
    response = spawn_random_woods.spawn_random_woods(spawn_proxy, delete_proxy, specs, replace=False)
    rospy.loginfo(
        "Respawned random_woods seed=%s total=%d success=%s",
        args.seed,
        len(specs),
        response.success,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("reset_random_woods_world failed: %s", exc)
        sys.exit(1)
