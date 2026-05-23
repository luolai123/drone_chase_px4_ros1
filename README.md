# ROS1 PX4 Gazebo RGB-D Drone Chase Simulation

基于 ROS1 Noetic、PX4 SITL、Gazebo Classic 和 MAVROS 的 RGB-D 无人机红球追捕仿真系统，采用规则专家、DAgger-lite/BC 和 PPO fine-tuning 的专家引导强化学习路线。

## System Architecture

Runtime closed loop:

```text
Gazebo RGB-D camera
  -> red_ball_detector
  -> depth_risk_estimator
  -> GazeboChaseEnv
  -> frozen policy
  -> safety_filter_node
  -> MAVROS/PX4
```

The final policy does not consume raw RGB-D tensors directly. RGB is converted into `/target/state` through HSV red-ball detection, and depth is converted into target-depth and obstacle-risk features. The policy input is a low-dimensional feature observation.

## Final Policy

Final frozen policy artifacts:

- model: `outputs/final_policy_registry/best_world0_world1_policy.zip`
- VecNormalize: `outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl`

The policy files are not intended to be committed directly to Git. Reproducibility is tracked through the final policy registry, SHA256 hashes, and reproduce commands.

## Validated Scenarios

- `world0`
- `world1` sparse obstacle
- `woods_easy`
- `random_woods`

## Not Validated

- dense woods
- `woods_hard`
- dynamic obstacles
- real RGB-D sensors
- real flight

## How To Run

Use the checked final command references:

- demo commands: `outputs/final_demo_commands.md`
- reproduction commands: `outputs/final_policy_registry/reproduce_commands.md`

The typical runtime stack uses Gazebo/PX4, Phase 3 perception, the Phase 6 safety runtime, and the frozen policy evaluation script. Keep the final frozen policy, action mapping, reward, and safety filter unchanged when reproducing the final validation.

## Requirements

Prepare the following before running the simulation:

- ROS Noetic
- Gazebo Classic
- MAVROS
- PX4-Autopilot v1.13.3
- Python environment with the project RL/perception dependencies

## Notes

- The current final policy uses the 20-dimensional observation v1 interface.
- Phase 8 designed a 28-dimensional observation v2 and built an offline dataset, but no observation_v2 policy has been trained.
- observation_v2 is for future Phase 9 work and must not be used with the current frozen policy.
