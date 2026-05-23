# Method Section For Paper

## 1. System Overview

We consider a simulated aerial target-chasing task in which a quadrotor must approach and capture a red spherical target while avoiding obstacles. The system is implemented in ROS1 Noetic with PX4 SITL, Gazebo Classic, and MAVROS. Gazebo provides the UAV model, RGB-D camera, red target, and obstacle environments. PX4 SITL provides the flight-control stack, and MAVROS bridges ROS velocity commands to PX4 OFFBOARD control.

The deployed closed-loop pipeline is feature-based rather than end-to-end visual control. RGB-D images are first converted into low-dimensional target and obstacle features by perception nodes. A frozen policy then maps these features to raw chase commands, and a safety filter applies runtime constraints before commands are sent to MAVROS/PX4.

## 2. RGB-D Perception

RGB-D sensing is decomposed into target perception and obstacle perception. The RGB stream is processed by HSV-based red-ball detection to estimate target visibility, image-space target center, target radius, and associated confidence/quality indicators. The aligned depth stream is used to estimate target depth near the detected target and to infer the target position in the camera frame.

For obstacle perception, the depth image is evaluated over forward and lateral regions of interest. The risk estimator computes robust low-percentile depth features, obstacle area ratio, and a binary/continuous danger signal. These outputs are published as ROS topics and consumed by the environment wrapper and safety filter.

The current final policy does not directly receive RGB images, depth images, point clouds, or learned CNN embeddings. It receives compact RGB-D-derived features.

## 3. Low-Dimensional Observation Construction

The final frozen policy uses observation v1, a 20-dimensional vector composed of target state, obstacle risk, UAV state, and previous-action terms. The target features come from `/target/state`, the obstacle features come from `/obstacle/risk`, vehicle height and velocity come from MAVROS state estimates, and previous-action features come from the policy-command history.

This design reduces policy input dimensionality and makes the control problem compatible with expert-guided imitation and conservative reinforcement-learning fine-tuning. It also enables direct diagnostic analysis of which perception features affect policy behavior. The cost is that the policy depends on the correctness and calibration of the hand-engineered perception interface.

## 4. Safety-Filtered Control

The policy output is not sent directly to PX4. Raw policy commands are first passed to `safety_filter_node.py`, which owns the final MAVROS setpoint publisher. The safety filter handles raw-command timeout protection, target-loss context, height limits, depth-stop behavior, emergency avoidance, velocity limiting, and body-to-world velocity conversion.

During final validation and Phase 8 documentation updates, the safety-filter control logic was kept fixed. This separation allows the learned policy to handle nominal pursuit behavior while the safety filter enforces runtime constraints in the simulated closed loop.

## 5. Expert-Guided Policy Learning

### Rule Expert

A rule-based expert was used to provide initial chase behavior and supervision. The expert encodes target pursuit and safety-aware heuristics and was used to bootstrap the dataset for imitation learning.

### DAgger-lite

Behavior cloning from the initial expert data suffered from closed-loop covariate shift. A DAgger-lite process collected additional expert labels on states visited by imperfect learned policies, improving coverage of states induced by the policy itself.

### BC v2

The aggregated dataset was used to train a second behavior-cloning policy. BC v2 provided a stable initialization and reduced the failure modes observed when training purely from the original expert dataset.

### PPO Fine-Tuning From BC

Conservative PPO fine-tuning was initialized from BC v2. The final checkpoint was selected by deterministic simulation validation rather than by training reward alone. The selected frozen policy remains tied to observation v1 and its corresponding VecNormalize statistics.

## 6. Feature Ablation and Observation v2 Design

Phase 8.2 performed diagnostic feature ablations on the frozen observation-v1 policy without retraining. The results indicate that the policy most strongly depends on camera-frame target position, image-space target center offsets, and forward obstacle-risk features. Velocity and previous-action terms appear less central and may be compressed in future designs.

Phase 8.3 designed observation v2 as a future 28-dimensional interface. It adds target confidence, detection-quality features, temporal target visibility/loss features, depth-validity features, smoothed target radius, five-sector depth features, and smoothed front depth. Observation v2 is not the input of the current final policy and has not been trained. It is a design artifact and offline dataset for potential Phase 9 training.

## 7. Final Deployment in Simulation

The final deployed simulation policy is the frozen observation-v1 policy stored in the final policy registry. It has been validated in world0, world1 sparse obstacle, woods_easy, and random_woods simulation scenarios. The validated scope does not include dense woods, woods_hard, dynamic obstacles, real RGB-D sensors, or real flight.

All final claims should be limited to the ROS1/PX4/Gazebo simulation stack and the validated scenario set. Real-world deployment would require additional perception validation, dynamics validation, hardware-in-the-loop testing, and flight safety review.
