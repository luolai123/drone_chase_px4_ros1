# Reproduce Commands

These commands reproduce evaluation only. They do not train, fine-tune, change reward, change `safety_filter_node.py`, change action mapping, change Phase 1/2/3 core logic, or replace the final registered policy artifacts.

## Final registered policy

```bash
MODEL=/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip
VECNORMALIZE=/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl
```

Load check:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
python3 - <<'PY'
from stable_baselines3 import PPO
model = PPO.load('/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip')
print(type(model).__name__)
PY
```

## Phase2 random_woods runtime

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=random_woods gui:=false interactive:=false
```

## Phase3 perception runtime

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

## Phase6 safety runtime

Terminal 3:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

## random_woods zero-shot eval

Terminal 4:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_random_woods_zeroshot_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world random_woods \
  --episodes 30 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/random_woods_zeroshot_from_final_policy
```

## random_woods robustness/stress eval

Terminal 4:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_random_woods_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world random_woods \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/random_woods_robustness_from_final_policy
```

## Other validated eval commands

World0 robustness:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world0_best_policy_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --episodes 30 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world0_best_policy_robustness
```

World1 sparse zero-shot:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world1_zeroshot_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --episodes 30 \
  --world world_1 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world1_zeroshot_from_registry
```

World1 sparse robustness/stress:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world1_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world world_1 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_world1_robustness_from_registry
```

Woods_easy zero-shot:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_woods_easy_zeroshot_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world woods_easy \
  --episodes 30 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_woods_easy_zeroshot_from_registry
```

Woods_easy robustness/stress:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_woods_easy_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world woods_easy \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_woods_easy_robustness_from_registry
```

Woods_easy reset gate:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_woods_easy_reset_fix_eval.py \
  --trials 50 \
  --output-dir /home/whk/vf_ws/outputs/phase7/repro_woods_easy_reset_fix
```

## Runtime checks

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /safety_filter/mode
rostopic echo -n 1 /target/state
rostopic echo -n 1 /obstacle/risk
rostopic echo -n 1 /mavros/setpoint_velocity/cmd_vel
```

## Cleanup

Use this after an eval run, especially before repeating random_woods robustness/stress:

```bash
pkill -f roslaunch || true
pkill -f gzserver || true
pkill -f gzclient || true
pkill -f px4 || true
pkill -f mavros || true
pkill -f safety_filter_node.py || true
pkill -f red_ball_detector.py || true
pkill -f depth_risk_estimator.py || true
pkill -f phase7_random_woods || true
```

## Reset pollution notes

- The first Phase 7.4B random_woods robustness attempt was polluted by Gazebo/reset failure and is archived at `/home/whk/vf_ws/outputs/phase7/random_woods_robustness_from_final_policy/attempt_reset_polluted_20260521_013011`.
- Final registry metrics use only the clean rerun artifacts in `/home/whk/vf_ws/outputs/phase7/random_woods_robustness_from_final_policy`.
- If `reset_pollution_detected=true`, OFFBOARD drops appear, or Gazebo crashes, archive that attempt separately and rerun from a clean ROS/Gazebo/PX4 process state.

## Validated boundary

Allowed claims:

- world0 passed
- world1 sparse obstacle passed
- woods_easy passed
- random_woods zero-shot passed
- random_woods robustness/stress passed

Disallowed claims:

- all woods passed
- dense woods passed
- woods_hard passed
- dynamic obstacles passed
- real hardware deployment passed
- real RGB-D perception passed
