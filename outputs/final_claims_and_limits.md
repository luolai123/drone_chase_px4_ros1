# Final Claims and Limits

## Allowed claims

- The ROS1/PX4/Gazebo RGB-D drone red-ball chase simulation system has been implemented.
- world0 has passed validation.
- world1 sparse obstacle has passed validation.
- woods_easy has passed validation.
- random_woods has passed validation.
- Safety intervention has been validated in the simulated gates, including front-obstacle intervention checks.
- The current frozen policy is reproducible from the final registry artifacts.
- Current RGB-D information is used in low-dimensional feature form through `/target/state` and `/obstacle/risk`, not as raw image tensors.
- Phase 8.2 analyzed key perception-feature dependencies of the frozen observation-v1 policy.
- Phase 8.3 completed observation_v2 design and offline observation_v2 dataset construction.

## Disallowed claims

- dense woods has passed.
- woods_hard has passed.
- dynamic obstacles have passed.
- Real hardware deployment has passed.
- Real RGB-D perception has passed.
- The policy already has real-drone generalization capability.
- An observation_v2 policy has been trained.
- observation_v2 is better than observation_v1.
- An end-to-end RGB-D policy has been implemented.
- Real RGB-D sensors have been validated.

## Exact validated scope

The validated scope is limited to ROS1/PX4/Gazebo simulation with the final frozen policy:

- world0 robustness: 30/30 success.
- world1 sparse zero-shot: 30/30 success.
- world1 sparse robustness/stress: 90/90 success.
- woods_easy zero-shot: 29/30 success.
- woods_easy robustness/stress: 90/90 success.
- woods_easy reset fix: 50/50 reset success.
- random_woods zero-shot: 29/30 success.
- random_woods robustness/stress: 88/90 success.

## Frozen policy

- model: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip`
- vecnormalize: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl`
- model_sha256: `f571329c999a303507d86e8a3ab19a9436fc32056e8ae171b81e0bac222f21b0`
- vecnormalize_sha256: `37a9cf28651731ba90593ce155bec5c717d113186a5a1fbe175978ab407b6033`

## Phase 8 observation notes

- observation_v1 is the active 20-dimensional input used by the final frozen policy.
- observation_v2 is a 28-dimensional future-design artifact only.
- observation_v2 has a schema, optional default-disabled builder, and offline dataset.
- observation_v2 has not been used to train or validate a policy.
- The current final policy must not be evaluated with observation_v2 or a mismatched VecNormalize file.
