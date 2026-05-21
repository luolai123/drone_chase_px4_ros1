#!/usr/bin/env python3

import argparse
import math
import random
import sys
import time

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


OBSTACLE_PREFIX = "obstacle_"
OBSTACLES_PARAM = "/drone_chase/obstacles"
RED_BALL_POSE_PARAM = "/drone_chase/red_ball_pose"


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def obstacle_sdf(name, spec):
    color = spec["color"]
    material = """
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
        </material>""".format(**color)

    if spec["shape"] == "box":
        geometry = """
          <box>
            <size>{sx:.3f} {sy:.3f} {height:.3f}</size>
          </box>""".format(**spec)
    else:
        geometry = """
          <cylinder>
            <radius>{radius:.3f}</radius>
            <length>{height:.3f}</length>
          </cylinder>""".format(**spec)

    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>{geometry}
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>{geometry}
        </geometry>{material}
      </visual>
    </link>
  </model>
</sdf>""".format(name=name, geometry=geometry, material=material)


def footprint_radius(spec):
    if spec["shape"] == "box":
        return 0.5 * math.hypot(spec["sx"], spec["sy"])
    return spec["radius"]


def sample_obstacle(rng):
    shape = rng.choice(["box", "cylinder"])
    height = rng.uniform(0.5, 2.5)
    if shape == "box":
        spec = {
            "shape": shape,
            "sx": rng.uniform(0.4, 1.2),
            "sy": rng.uniform(0.4, 1.2),
            "height": height,
        }
    else:
        spec = {
            "shape": shape,
            "radius": rng.uniform(0.25, 0.7),
            "height": height,
        }

    shade = rng.uniform(0.25, 0.55)
    spec["color"] = {"r": shade, "g": shade, "b": shade}
    return spec


def is_valid_xy(x, y, radius, specs, red_ball_xy):
    if math.hypot(x, y) < 1.4 + radius:
        return False
    if red_ball_xy and math.hypot(x - red_ball_xy[0], y - red_ball_xy[1]) < 1.0 + radius:
        return False
    for other in specs:
        other_radius = footprint_radius(other)
        if math.hypot(x - other["x"], y - other["y"]) < radius + other_radius + 0.45:
            return False
    return True


def generate_obstacle_specs(num, world_size, rng, red_ball_xy=None):
    specs = []
    half = float(world_size) / 2.0
    for _ in range(int(num)):
        placed = False
        for _attempt in range(300):
            spec = sample_obstacle(rng)
            radius = footprint_radius(spec)
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
            if not is_valid_xy(x, y, radius, specs, red_ball_xy):
                continue
            spec["x"] = x
            spec["y"] = y
            spec["z"] = spec["height"] / 2.0
            specs.append(spec)
            placed = True
            break
        if not placed:
            rospy.logwarn("Could not place one obstacle without violating spacing constraints")
    return specs


def normalize_seed(seed):
    if seed is None or seed < 0:
        return None
    return seed


def get_red_ball_xy(timeout_sec=10.0):
    deadline = time.time() + timeout_sec
    while not rospy.is_shutdown() and time.time() < deadline:
        if rospy.has_param(RED_BALL_POSE_PARAM):
            pose = rospy.get_param(RED_BALL_POSE_PARAM)
            return float(pose["x"]), float(pose["y"])
        rospy.sleep(0.1)
    rospy.logwarn("No %s param found; using x=4.0 y=0.0 for obstacle spacing", RED_BALL_POSE_PARAM)
    return 4.0, 0.0


def delete_existing_obstacles(delete_proxy, world_proxy):
    try:
        names = world_proxy().model_names
    except rospy.ServiceException:
        names = []
    for name in names:
        if name.startswith(OBSTACLE_PREFIX):
            try:
                delete_proxy(name)
            except rospy.ServiceException:
                pass


def spawn_obstacles(spawn_proxy, specs):
    obstacle_params = []
    for index, spec in enumerate(specs):
        name = "{}{}".format(OBSTACLE_PREFIX, index)
        response = spawn_proxy(name, obstacle_sdf(name, spec), "", make_pose(spec["x"], spec["y"], spec["z"]), "world")
        if not response.success:
            raise RuntimeError("{}: {}".format(name, response.status_message))

        record = {
            "name": name,
            "shape": spec["shape"],
            "x": float(spec["x"]),
            "y": float(spec["y"]),
            "z": float(spec["z"]),
            "height": float(spec["height"]),
        }
        if spec["shape"] == "box":
            record.update({"sx": float(spec["sx"]), "sy": float(spec["sy"])})
        else:
            record.update({"radius": float(spec["radius"])})
        obstacle_params.append(record)
        rospy.loginfo(
            "Spawned %s shape=%s x=%.3f y=%.3f z=%.3f height=%.3f",
            name,
            spec["shape"],
            spec["x"],
            spec["y"],
            spec["z"],
            spec["height"],
        )

    rospy.set_param(OBSTACLES_PARAM, obstacle_params)
    return obstacle_params


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Spawn random sparse obstacles in Gazebo.")
    parser.add_argument("--num", type=int, default=4, help="Number of obstacles to spawn.")
    parser.add_argument("--world-size", type=float, default=12.0, help="Square world side length in meters.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--red-ball-x", type=float, default=None, help="Known red ball x; otherwise ROS param is used.")
    parser.add_argument("--red-ball-y", type=float, default=None, help="Known red ball y; otherwise ROS param is used.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete existing obstacle_* models first.")
    return parser


def main():
    args = build_arg_parser().parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("spawn_random_obstacles")

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/get_world_properties")
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)

    if not args.keep_existing:
        delete_existing_obstacles(delete_proxy, world_proxy)
        rospy.sleep(0.2)

    if args.red_ball_x is not None and args.red_ball_y is not None:
        red_ball_xy = (args.red_ball_x, args.red_ball_y)
    else:
        red_ball_xy = get_red_ball_xy()

    specs = generate_obstacle_specs(args.num, args.world_size, random.Random(normalize_seed(args.seed)), red_ball_xy)
    spawn_obstacles(spawn_proxy, specs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("spawn_random_obstacles failed: %s", exc)
        sys.exit(1)
