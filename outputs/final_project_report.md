# Final Project Report

## 1. Project background

This project implements a ROS1/PX4/Gazebo RGB-D drone red-ball chase simulation system. The goal is to evaluate a frozen UAV chase policy in closed-loop simulation across progressively harder environments, while keeping the safety filter and action mapping fixed during final validation.

The final frozen policy is:

- model: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip`
- vecnormalize: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl`
- selected checkpoint: `ppo_step_2500`

## 2. System architecture

The system is organized as a closed-loop simulation stack:

- Gazebo provides the world, RGB-D camera, red target, obstacles, and UAV model.
- PX4 SITL runs the flight controller.
- MAVROS bridges ROS velocity commands to PX4 OFFBOARD control.
- Phase 3 perception extracts red-ball target state and depth-risk estimates from RGB-D streams.
- `safety_filter_node.py` is the only runtime publisher to MAVROS velocity setpoints.
- `GazeboChaseEnv` wraps ROS topics, reset logic, observations, rewards, and policy actions for evaluation/training scripts.
- Final policy evaluation scripts load the frozen PPO policy and frozen VecNormalize statistics.

## 3. ROS/PX4/Gazebo/MAVROS stack

Runtime uses ROS Noetic, PX4 SITL, Gazebo, and MAVROS. The main launch flow is:

- `phase2_chase_world.launch` starts Gazebo, PX4 SITL, MAVROS, the UAV model, and the selected world.
- `phase3_perception.launch` starts red-ball detection and depth-risk estimation.
- `phase6_env_runtime.launch` starts the safety-filter runtime.
- Phase 7 evaluation scripts load the final registered model and publish raw policy commands into the safety-filter loop.

## 4. Perception module

The perception stack contains:

- `red_ball_detector.py`: detects the red target from RGB and estimates target depth/position using aligned RGB-D information.
- `depth_risk_estimator.py`: computes depth-risk features such as front/left/right q05 depth, obstacle area ratio, and obstacle danger.

The final low-dimensional policy uses target geometry, depth-risk features, vehicle state, and previous-action signals through `GazeboChaseEnv`.

The RGB-D stream is not consumed by the frozen policy as raw images. RGB images are processed by HSV red-ball detection to publish `/target/state`; depth images are used for target depth estimation and obstacle-risk estimation. The policy therefore follows an RGB-D feature-based expert-guided RL design, not an end-to-end RGB-D RL design.

## 4.1 RGB-D feature interface

The deployed simulation stack uses RGB-D as follows:

- RGB image: HSV segmentation detects the red ball, estimates image-space target center/radius, and publishes target visibility and target geometry through `/target/state`.
- Depth image for target: depth samples around the detected red ball estimate target depth and camera-frame target position.
- Depth image for obstacles: depth regions of interest estimate front/left/right obstacle proximity, area ratio, and danger through `/obstacle/risk`.
- Policy input: the frozen policy receives low-dimensional perception features, MAVROS state, and previous-action terms; it does not receive RGB-D tensors.

This distinction matters for interpretation: the final policy validates a perception-feature control policy under RGB-D sensing, not an end-to-end visual policy.

## 4.2 Observation v1 used by the frozen policy

The current final frozen policy uses the original 20-dimensional observation vector:

| index | field | primary source |
| --- | --- | --- |
| 0 | `target_visible` | `/target/state` |
| 1 | `target_x_camera` | `/target/state` |
| 2 | `target_y_camera` | `/target/state` |
| 3 | `target_z_camera` | `/target/state` |
| 4 | `target_distance` | `/target/state` |
| 5 | `target_u` | `/target/state` |
| 6 | `target_v` | `/target/state` |
| 7 | `front_q05_depth` | `/obstacle/risk` |
| 8 | `left_q05_depth` | `/obstacle/risk` |
| 9 | `right_q05_depth` | `/obstacle/risk` |
| 10 | `obstacle_area_ratio` | `/obstacle/risk` |
| 11 | `obstacle_danger` | `/obstacle/risk` |
| 12 | `drone_vx` | MAVROS state |
| 13 | `drone_vy` | MAVROS state |
| 14 | `drone_vz` | MAVROS state |
| 15 | `drone_z` | MAVROS pose |
| 16 | `prev_vx` | previous action |
| 17 | `prev_vy` | previous action |
| 18 | `prev_vz` | previous action |
| 19 | `prev_yaw_rate` | previous action |

`GazeboChaseEnv` keeps this 20-dimensional observation as the default runtime interface. The final policy and its VecNormalize statistics are tied to this schema and must not be used with a different observation layout.

## 4.3 Phase 8 feature ablation summary

Phase 8.2 evaluated observation feature ablations without retraining or modifying the frozen policy. The main findings were:

- The top-3 critical feature groups are `target_x/y/z_camera`, `target_u/v`, and obstacle-risk features (`front_q05_depth`, `obstacle_area_ratio`, `obstacle_danger`).
- Target camera-frame position and image-space center offsets are central to chase direction, yaw centering, and target approach.
- Front depth and obstacle danger explain safety-filter interventions and collision risk in woods/random_woods layouts.
- Velocity and previous-action terms appear less critical than target and obstacle features and can be considered for compression in a future observation design.
- Detection-quality signals and depth-sector signals have diagnostic value for target-loss and safety-context analysis.

These results support a future observation redesign but do not change the current final policy.

## 4.4 Observation v2 design status

Phase 8.3 produced an `observation_v2` design for future Phase 9 work. It is a 28-dimensional schema and is not used by the current frozen policy.

The proposed additions include:

- target confidence;
- detection quality score;
- target visible ratio over a window;
- normalized target lost frames;
- target depth valid ratio;
- smoothed target radius;
- five-sector depth q05 features;
- smoothed front q05 depth.

The observation_v2 builder is implemented as an optional, default-disabled Python module. An offline dataset has been built for analysis, but no observation_v2 policy has been trained or validated. Any future observation_v2 policy requires new training artifacts, new VecNormalize statistics, and a separate registry under `outputs/phase9/`.

## 5. safety_filter_node

`safety_filter_node.py` owns the final command path to MAVROS. It handles:

- PX4 connection, OFFBOARD mode, arming, and takeoff state machine.
- Raw command timeout protection.
- Target lost behavior.
- Height constraints.
- Depth stop and emergency avoidance behavior.
- Velocity limiting and body-to-world command publishing.

During final phases, this node was not modified. Safety intervention was explicitly validated in world1, woods_easy, and random_woods front-obstacle intervention gates.

## 6. GazeboChaseEnv

`GazeboChaseEnv` provides the gym-style interface used by Phase 7 scripts. It:

- reads target, depth-risk, MAVROS, and safety-filter topics;
- maps policy actions to raw body-frame commands;
- manages Gazebo/PX4 resets;
- computes episode termination and logging fields;
- records success, timeout, collision, out-of-bounds, height violation, OFFBOARD drop, and reset-pollution indicators.

The final evaluation uses the same frozen action mapping and frozen VecNormalize statistics.

## 7. Rule expert

The rule expert provided initial closed-loop chase behavior and supervision data. It was used to bootstrap behavior cloning and DAgger-lite data collection, especially for states visited by imperfect policies.

## 8. PPO from scratch failure

Pure PPO from scratch was unstable in this setting. The main issues were sparse/fragile closed-loop feedback, reset sensitivity, safety-filter interactions, and exploration that could find degenerate behaviors before learning reliable target pursuit.

## 9. Reward hacking fix

Reward hacking was audited and fixed before selecting the final policy. The final path avoided selecting policies that exploited reward terms without robust chase behavior. Final acceptance was based on deterministic closed-loop validation gates, not training reward alone.

## 10. DAgger-lite + BC v2

BC v1 exposed closed-loop covariate shift: the model could imitate expert-like states but drift under its own state distribution. DAgger-lite collected additional expert labels on BC-visited states and built the BC v2 dataset. BC v2 became the stable initialization for later PPO fine-tuning.

## 11. PPO fine-tuning from BC v2

Conservative PPO fine-tuning from BC v2 succeeded. Checkpoint `ppo_step_2500` was selected by deterministic sweep and registered as the final frozen policy.

## 12. Validation results

| gate | result | success_rate |
| --- | --- | --- |
| world0 robustness | 30/30 | 1.0000 |
| world1 sparse zero-shot | 30/30 | 1.0000 |
| world1 sparse robustness/stress | 90/90 | 1.0000 |
| woods_easy zero-shot | 29/30 | 0.9667 |
| woods_easy robustness/stress | 90/90 | 1.0000 |
| woods_easy reset fix | 50/50 | 1.0000 |
| random_woods zero-shot | 29/30 | 0.9667 |
| random_woods robustness/stress | 88/90 | 0.9778 |

Random_woods robustness/stress also passed Group D safety intervention. The first polluted random_woods robustness attempt was archived and excluded from the final registry.

## 13. Current limitations

The final registry validates world0, world1 sparse obstacle, woods_easy, and random_woods only. It does not validate:

- dense woods;
- woods_hard;
- dynamic obstacles;
- real RGB-D perception;
- real flight or hardware deployment;
- real-world UAV generalization.

## 14. Future work

Recommended next steps:

- update documentation and method descriptions to clearly distinguish feature-based RGB-D policy input from end-to-end RGB-D RL;
- use the Phase 8.2/8.3 results to design Phase 9 experiments without overwriting the current final frozen policy;
- run separate dense woods validation;
- run separate woods_hard validation;
- design dynamic obstacle gates;
- add real RGB-D dataset/perception validation;
- perform hardware-in-the-loop or real-flight safety validation before any deployment claim.
