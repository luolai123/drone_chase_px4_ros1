# Final Phase 7 World0/World1 Report

This legacy filename is retained for compatibility. The more accurate final report after Phase 7.3C is:

`final_phase7_world0_world1_woods_easy_report.md`

## 1. Project scope

This registry freezes the current best policy and records validation for:

- world0 red-ball chase robustness
- world1 sparse-obstacle zero-shot and robustness/stress
- woods_easy zero-shot, robustness/stress, and reset gate

It does not include random_woods, dense woods, woods_hard, dynamic obstacles, real hardware deployment, or real RGB-D perception claims.

## 2. Final policy

- policy_name: world0_world1_sparse_best_ppo_from_bc_v2_step2500
- model: /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip
- vecnormalize: /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl
- checkpoint: ppo_step_2500
- Phase 7.3C policy status: frozen; no training, fine-tuning, reward change, safety-filter control change, or action-mapping change.

## 3. Training route

- Pure PPO failed to provide the required stable final behavior.
- Reward v2 fixed the observed reward-hacking behavior.
- BC v1 failed online because of covariate shift.
- DAgger-lite + BC v2 succeeded as the imitation-learning base.
- PPO from BC v2 succeeded; deterministic checkpoint sweep selected ppo_step_2500.
- Phase 7.3C only updated the registry with completed woods_easy validation results.

## 4. Validated claims

- ROS1/PX4/Gazebo RGB-D drone red-ball chase simulation system has been implemented.
- Rule expert, DAgger/BC, PPO fine-tuning, and safety-filter closed loop have been completed.
- Final registered policy has passed world0 validation.
- Final registered policy has passed world1 sparse obstacle validation.
- Final registered policy has passed woods_easy zero-shot and robustness/stress validation.

## 5. World0 validation

- episodes: 30
- success_rate: 1.0000 (30/30)
- target_visible_ratio_mean: 1.0000
- final_distance_mean: 0.7836 m
- min_distance_mean: 0.7834 m
- RAW_TIMEOUT: 0
- reset_pollution_detected: no

## 6. World1 sparse obstacle validation

- zero-shot episodes: 30
- zero-shot success_rate: 1.0000 (30/30)
- robustness/stress episodes: 90
- robustness/stress success_rate: 1.0000 (90/90)
- Group A/B/C/D success rates: 1.0000 / 1.0000 / 1.0000 / 1.0000
- collision/out_of_bounds/height_violation: 0 / 0 / 0
- OFFBOARD drop count: 0

## 7. Woods_easy validation

- zero-shot episodes: 30
- zero-shot success_rate: 0.9667 (29/30)
- zero-shot collision/emergency failure rate: 0.0333 (1/30)
- robustness/stress episodes: 90
- robustness/stress success_rate: 1.0000 (90/90)
- robustness/stress collision/emergency failure rate: 0.0000 (0/90)
- reset-only trials: 50
- reset_success_rate: 1.0000 (50/50)
- reset_pollution_detected: false

## 8. Safety intervention result

- world1 front-obstacle safety intervention: passed
- woods_easy Group D safety intervention: passed
- danger_seen: yes
- safety_triggered: yes
- filtered_vx_body <= 0: yes
- no_collision: yes
- no_offboard: yes
- recovered_or_safe: yes

## 9. Policy artifacts

- policy_manifest.json
- best_world0_world1_policy.zip
- best_world0_world1_vecnormalize.pkl
- final_results_table.csv
- final_results_table.md
- reproduce_commands.md
- final_phase7_world0_world1_woods_easy_report.md
- phase7_3c_registry_update_report.md

## 10. Boundaries

Do not claim:

- random_woods has passed
- dense woods has passed
- woods_hard has passed
- dynamic obstacles have passed
- real hardware deployment has passed
- real RGB-D perception has passed

## 11. Next phase recommendation

It is allowed to enter random_woods zero-shot evaluation as a new validation phase. It is not allowed to claim all woods scenarios have passed.
