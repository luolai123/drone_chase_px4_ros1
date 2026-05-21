#!/usr/bin/env python3

import argparse
import math
import random
import sys

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


MODEL_NAME = "random_woods"
WOODS_PARAM = "/drone_chase/random_woods"

GROUND_MARGIN = 0.02

DEFAULTS = {
    "seed": 42,
    "area_x_min": 1.0,
    "area_x_max": 6.5,
    "area_y_min": -3.5,
    "area_y_max": 3.5,
    "num_trunks": 18,
    "num_branches": 45,
    "num_fallen": 10,
    "uav_x": 0.0,
    "uav_y": 0.0,
    "uav_clearance": 1.0,
    "target_x": 4.0,
    "target_y": 0.0,
    "target_clearance": 0.6,
    "max_attempts": 500,
}


def normalize_seed(seed):
    if seed is None or seed < 0:
        return None
    return int(seed)


def make_pose(x=0.0, y=0.0, z=0.0):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def direction_from_tilt_yaw(tilt, yaw):
    """Cylinder local +Z becomes this world direction after rpy=(0, tilt, yaw)."""
    return (
        math.sin(tilt) * math.cos(yaw),
        math.sin(tilt) * math.sin(yaw),
        math.cos(tilt),
    )


def cylinder_endpoints(spec):
    dx, dy, dz = direction_from_tilt_yaw(spec["pitch"], spec["yaw"])
    half_len = 0.5 * spec["length"]
    return (
        (
            spec["x"] - half_len * dx,
            spec["y"] - half_len * dy,
            spec["z"] - half_len * dz,
        ),
        (
            spec["x"] + half_len * dx,
            spec["y"] + half_len * dy,
            spec["z"] + half_len * dz,
        ),
    )


def point_segment_distance_xy(px, py, start, end):
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    vx = ex - sx
    vy = ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * vx + (py - sy) * vy) / denom
    t = max(0.0, min(1.0, t))
    cx = sx + t * vx
    cy = sy + t * vy
    return math.hypot(px - cx, py - cy)


def minimum_center_z(length, radius, tilt):
    # The cylinder axis endpoints stay above ground, and the tilted radius is also kept clear.
    return GROUND_MARGIN + 0.5 * length * abs(math.cos(tilt)) + radius * abs(math.sin(tilt))


def sample_kind_spec(kind, rng):
    if kind == "trunk":
        radius = rng.uniform(0.04, 0.10)
        length = rng.uniform(1.0, 2.5)
        tilt = math.radians(rng.uniform(0.0, 25.0))
        z = minimum_center_z(length, radius, tilt)
    elif kind == "branch":
        radius = rng.uniform(0.02, 0.05)
        length = rng.uniform(0.8, 2.5)
        tilt = math.radians(rng.uniform(25.0, 80.0))
        z = max(rng.uniform(0.3, 2.2), minimum_center_z(length, radius, tilt))
    elif kind == "fallen":
        radius = rng.uniform(0.03, 0.08)
        length = rng.uniform(1.0, 3.5)
        tilt = math.radians(rng.uniform(70.0, 90.0))
        z = max(rng.uniform(0.15, 0.8), minimum_center_z(length, radius, tilt))
    else:
        raise ValueError("Unknown woods element kind: {}".format(kind))

    return {
        "kind": kind,
        "radius": radius,
        "length": length,
        "roll": 0.0,
        "pitch": tilt,
        "yaw": rng.uniform(-math.pi, math.pi),
        "z": z,
    }


def is_valid_clearance(spec, uav_x, uav_y, uav_clearance, target_x, target_y, target_clearance):
    start, end = cylinder_endpoints(spec)
    radius = spec["radius"]
    if point_segment_distance_xy(uav_x, uav_y, start, end) < uav_clearance + radius:
        return False
    if point_segment_distance_xy(target_x, target_y, start, end) < target_clearance + radius:
        return False
    return True


def generate_woods_specs(
    seed=DEFAULTS["seed"],
    area_x_min=DEFAULTS["area_x_min"],
    area_x_max=DEFAULTS["area_x_max"],
    area_y_min=DEFAULTS["area_y_min"],
    area_y_max=DEFAULTS["area_y_max"],
    num_trunks=DEFAULTS["num_trunks"],
    num_branches=DEFAULTS["num_branches"],
    num_fallen=DEFAULTS["num_fallen"],
    uav_x=DEFAULTS["uav_x"],
    uav_y=DEFAULTS["uav_y"],
    uav_clearance=DEFAULTS["uav_clearance"],
    target_x=DEFAULTS["target_x"],
    target_y=DEFAULTS["target_y"],
    target_clearance=DEFAULTS["target_clearance"],
    max_attempts=DEFAULTS["max_attempts"],
):
    rng = random.Random(normalize_seed(seed))
    specs = []
    requested = [
        ("trunk", int(num_trunks)),
        ("branch", int(num_branches)),
        ("fallen", int(num_fallen)),
    ]

    for kind, count in requested:
        placed_for_kind = 0
        for requested_index in range(count):
            placed = False
            for _attempt in range(int(max_attempts)):
                spec = sample_kind_spec(kind, rng)
                spec["x"] = rng.uniform(float(area_x_min), float(area_x_max))
                spec["y"] = rng.uniform(float(area_y_min), float(area_y_max))
                if not is_valid_clearance(spec, uav_x, uav_y, uav_clearance, target_x, target_y, target_clearance):
                    continue
                spec["name"] = "{}_{:03d}".format(kind, requested_index)
                specs.append(spec)
                placed_for_kind += 1
                placed = True
                break
            if not placed:
                rospy.logwarn(
                    "Could not place %s %d after %d attempts",
                    kind,
                    requested_index,
                    max_attempts,
                )
        if placed_for_kind < count:
            rospy.logwarn("Placed %d/%d requested %s elements", placed_for_kind, count, kind)

    return specs


def material_xml():
    return """        <material>
          <ambient>0.45 0.22 0.05 1</ambient>
          <diffuse>0.65 0.32 0.08 1</diffuse>
          <specular>0.08 0.06 0.04 1</specular>
        </material>"""


def cylinder_geometry_xml(spec):
    return """          <cylinder>
            <radius>{radius:.4f}</radius>
            <length>{length:.4f}</length>
          </cylinder>""".format(**spec)


def link_sdf(spec):
    geometry = cylinder_geometry_xml(spec)
    return """    <link name="{name}">
      <pose>{x:.4f} {y:.4f} {z:.4f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
      <collision name="collision">
        <geometry>
{geometry}
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
{geometry}
        </geometry>
{material}
      </visual>
    </link>""".format(
        name=spec["name"],
        x=spec["x"],
        y=spec["y"],
        z=spec["z"],
        roll=spec["roll"],
        pitch=spec["pitch"],
        yaw=spec["yaw"],
        geometry=geometry,
        material=material_xml(),
    )


def woods_sdf(specs):
    links = "\n".join(link_sdf(spec) for spec in specs)
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <self_collide>false</self_collide>
{links}
  </model>
</sdf>""".format(model_name=MODEL_NAME, links=links)


def spec_record(spec):
    start, end = cylinder_endpoints(spec)
    return {
        "name": spec["name"],
        "kind": spec["kind"],
        "x": float(spec["x"]),
        "y": float(spec["y"]),
        "z": float(spec["z"]),
        "radius": float(spec["radius"]),
        "length": float(spec["length"]),
        "roll": float(spec["roll"]),
        "pitch": float(spec["pitch"]),
        "yaw": float(spec["yaw"]),
        "start": [float(start[0]), float(start[1]), float(start[2])],
        "end": [float(end[0]), float(end[1]), float(end[2])],
    }


def count_specs(specs, kind):
    return sum(1 for spec in specs if spec["kind"] == kind)


def model_exists(world_proxy, model_name):
    try:
        return model_name in world_proxy().model_names
    except rospy.ServiceException:
        return False


def delete_random_woods(delete_proxy, world_proxy=None):
    if world_proxy is not None and not model_exists(world_proxy, MODEL_NAME):
        rospy.loginfo("No existing %s model to delete", MODEL_NAME)
        return None
    try:
        response = delete_proxy(MODEL_NAME)
        if response.success:
            rospy.loginfo("Deleted existing %s", MODEL_NAME)
        return response
    except rospy.ServiceException:
        return None


def spawn_random_woods(spawn_proxy, delete_proxy, specs, replace=True, world_proxy=None):
    if replace:
        delete_random_woods(delete_proxy, world_proxy=world_proxy)
        rospy.sleep(0.2)

    response = spawn_proxy(MODEL_NAME, woods_sdf(specs), "", make_pose(), "world")
    if not response.success:
        raise RuntimeError(response.status_message)

    rospy.set_param(WOODS_PARAM, [spec_record(spec) for spec in specs])
    return response


def validate_args(args, parser):
    if args.area_x_min >= args.area_x_max:
        parser.error("--area-x-min must be smaller than --area-x-max")
    if args.area_y_min >= args.area_y_max:
        parser.error("--area-y-min must be smaller than --area-y-max")
    if args.num_trunks < 0 or args.num_branches < 0 or args.num_fallen < 0:
        parser.error("woods element counts must be non-negative")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")


def log_summary(args, specs, spawned=None):
    total = len(specs)
    lines = [
        "random_woods seed={}".format(args.seed),
        "random_woods requested trunks={} branches={} fallen={}".format(
            args.num_trunks,
            args.num_branches,
            args.num_fallen,
        ),
        "random_woods generated total={} trunks={} branches={} fallen={}".format(
            total,
            count_specs(specs, "trunk"),
            count_specs(specs, "branch"),
            count_specs(specs, "fallen"),
        ),
        "random_woods area x=[{:.3f}, {:.3f}] y=[{:.3f}, {:.3f}]".format(
            args.area_x_min,
            args.area_x_max,
            args.area_y_min,
            args.area_y_max,
        ),
        "random_woods uav=({:.3f}, {:.3f}) uav_clearance={:.3f} target=({:.3f}, {:.3f}) target_clearance={:.3f}".format(
            args.uav_x,
            args.uav_y,
            args.uav_clearance,
            args.target_x,
            args.target_y,
            args.target_clearance,
        ),
    ]
    if spawned is not None:
        lines.append("random_woods spawn_success={}".format(bool(spawned)))

    if rospy.core.is_initialized():
        for line in lines:
            rospy.loginfo(line)
    else:
        for line in lines:
            print(line, file=sys.stderr)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Spawn a compound random woods SDF model in Gazebo.")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Random seed; use a negative value for nondeterministic.")
    parser.add_argument("--area-x-min", type=float, default=DEFAULTS["area_x_min"], help="Minimum obstacle center x.")
    parser.add_argument("--area-x-max", type=float, default=DEFAULTS["area_x_max"], help="Maximum obstacle center x.")
    parser.add_argument("--area-y-min", type=float, default=DEFAULTS["area_y_min"], help="Minimum obstacle center y.")
    parser.add_argument("--area-y-max", type=float, default=DEFAULTS["area_y_max"], help="Maximum obstacle center y.")
    parser.add_argument("--num-trunks", type=int, default=DEFAULTS["num_trunks"], help="Number of vertical or slightly tilted trunks.")
    parser.add_argument("--num-branches", type=int, default=DEFAULTS["num_branches"], help="Number of tilted branches.")
    parser.add_argument("--num-fallen", type=int, default=DEFAULTS["num_fallen"], help="Number of near-horizontal fallen logs.")
    parser.add_argument("--uav-x", type=float, default=DEFAULTS["uav_x"], help="UAV x used for XY clearance.")
    parser.add_argument("--uav-y", type=float, default=DEFAULTS["uav_y"], help="UAV y used for XY clearance.")
    parser.add_argument("--uav-clearance", type=float, default=DEFAULTS["uav_clearance"], help="XY clearance around the UAV position.")
    parser.add_argument("--target-x", type=float, default=DEFAULTS["target_x"], help="Red ball x used for clearance.")
    parser.add_argument("--target-y", type=float, default=DEFAULTS["target_y"], help="Red ball y used for clearance.")
    parser.add_argument("--target-clearance", type=float, default=DEFAULTS["target_clearance"], help="XY clearance around the red ball.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULTS["max_attempts"], help="Placement attempts per woods element.")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated SDF and do not call Gazebo.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    validate_args(args, parser)

    specs = generate_woods_specs(
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

    if args.dry_run:
        log_summary(args, specs, spawned=False)
        print(woods_sdf(specs))
        return

    rospy.init_node("spawn_random_woods")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/get_world_properties")
    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    world_proxy = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)

    response = spawn_random_woods(spawn_proxy, delete_proxy, specs, replace=True, world_proxy=world_proxy)
    log_summary(args, specs, spawned=response.success)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if rospy.core.is_initialized():
            rospy.logerr("spawn_random_woods failed: %s", exc)
        else:
            print("spawn_random_woods failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
