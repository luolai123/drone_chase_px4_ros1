# Architecture Diagram Description

## Figure 1: Closed-Loop System Architecture

### Nodes

- Gazebo world and PX4 SITL
- Simulated UAV with RGB-D camera
- `red_ball_detector`
- `depth_risk_estimator`
- `GazeboChaseEnv`
- Frozen policy
- `safety_filter_node`
- MAVROS
- PX4 OFFBOARD controller

### Arrows

```text
Gazebo/PX4 SITL
  -> RGB-D camera
  -> perception nodes
  -> GazeboChaseEnv
  -> frozen policy
  -> safety_filter_node
  -> MAVROS
  -> PX4
  -> Gazebo/PX4 SITL
```

### Caption

The closed-loop simulation converts Gazebo RGB-D sensor streams into low-dimensional perception features, evaluates the frozen policy, filters commands through runtime safety logic, and sends velocity setpoints to PX4 through MAVROS.

## Figure 2: RGB-D Information Flow

### Nodes

- RGB image
- HSV red-ball detection
- Target center `u/v`
- Target radius
- Target confidence and detection quality
- Depth image
- Target depth estimator
- Obstacle depth-sector estimator
- Low-dimensional observation

### Arrows

```text
RGB
  -> HSV red-ball detection
  -> target u/v/radius/confidence
  -> target-state features
  -> low-dimensional observation

Depth
  -> target depth estimation
  -> target camera-frame position
  -> low-dimensional observation

Depth
  -> obstacle risk sectors
  -> front/side q05, area ratio, danger
  -> low-dimensional observation
```

### Caption

RGB-D data is used as structured perception input. The current policy is not end-to-end RGB-D; it receives target and obstacle features derived from RGB-D perception nodes.

## Figure 3: Expert-Guided Training Route

### Nodes

- Rule expert
- DAgger-lite data collection
- BC v2
- PPO fine-tuning from BC
- Frozen final policy
- Multi-scenario validation

### Arrows

```text
Rule expert
  -> DAgger-lite demos
  -> BC v2
  -> PPO fine-tuning
  -> frozen policy
  -> multi-scenario validation
```

### Caption

Training follows an expert-guided route. Rule expert behavior bootstraps imitation data, DAgger-lite reduces closed-loop covariate shift, BC v2 initializes the policy, and PPO fine-tuning produces the frozen policy selected by deterministic validation.
