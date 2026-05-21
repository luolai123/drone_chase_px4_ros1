# Final Demo Commands

These commands are for evaluation/demo only. They do not train, fine-tune, change reward, change `safety_filter_node.py`, change action mapping, or replace the final policy.

## 1. Environment

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
```

## 2. Start Phase2 worlds

World0:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=world_0 gui:=false num_obstacles:=0
```

World1 sparse:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=world_1 gui:=false num_obstacles:=4
```

Woods_easy:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=woods_easy gui:=false interactive:=false
```

Random_woods:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
source /home/whk/vf_ws/src/drone_chase/scripts/source_px4_gazebo_env.sh
roslaunch drone_chase phase2_chase_world.launch world:=random_woods gui:=false interactive:=false
```

## 3. Start Phase3 perception

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase3_perception.launch debug:=false
```

## 4. Start Phase6 safety runtime

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
roslaunch drone_chase phase6_env_runtime.launch
```

## 5. Load final policy eval

Final policy artifacts:

```bash
MODEL=/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip
VECNORMALIZE=/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl
```

World0 robustness:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world0_best_policy_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --episodes 30 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/demo_world0_robustness
```

World1 zero-shot:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_world1_zeroshot_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --episodes 30 \
  --world world_1 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/demo_world1_zeroshot
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
  --output-dir /home/whk/vf_ws/outputs/phase7/demo_woods_easy_zeroshot
```

Random_woods zero-shot:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_random_woods_zeroshot_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world random_woods \
  --episodes 30 \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/demo_random_woods_zeroshot
```

Random_woods robustness/stress:

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rosrun drone_chase phase7_random_woods_robustness_eval.py \
  --model /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip \
  --vecnormalize /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl \
  --world random_woods \
  --deterministic \
  --output-dir /home/whk/vf_ws/outputs/phase7/demo_random_woods_robustness
```

## 6. Common rostopic checks

```bash
source /opt/ros/noetic/setup.bash
source /home/whk/vf_ws/devel/setup.bash
rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /safety_filter/mode
rostopic echo -n 1 /target/state
rostopic echo -n 1 /obstacle/risk
rostopic echo -n 1 /raw_cmd_vel
rostopic echo -n 1 /mavros/setpoint_velocity/cmd_vel
```

## 7. Cleanup

```bash
pkill -f roslaunch || true
pkill -f gzserver || true
pkill -f gzclient || true
pkill -f px4 || true
pkill -f mavros || true
pkill -f safety_filter_node.py || true
pkill -f red_ball_detector.py || true
pkill -f depth_risk_estimator.py || true
pkill -f phase7 || true
```
