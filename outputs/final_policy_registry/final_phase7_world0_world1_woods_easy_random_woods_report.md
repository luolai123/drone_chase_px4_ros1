# Final Phase 7 World0/World1/Woods_easy/Random_woods Report

## 1. Project scope

This report records the current frozen-policy validation state for the ROS1/PX4/Gazebo RGB-D drone red-ball chase simulation system. The validated simulation scope is world0, world1 sparse obstacle, woods_easy, and random_woods.

Phase 7.4C is a registry/report update only. No training, fine-tuning, reward change, safety-filter control change, action-mapping change, Phase 1/2/3 logic change, model replacement, or VecNormalize replacement was performed.

## 2. Final frozen policy

- policy_name: world0_world1_sparse_best_ppo_from_bc_v2_step2500
- model: /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip
- vecnormalize: /home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl
- checkpoint: ppo_step_2500
- model_sha256: f571329c999a303507d86e8a3ab19a9436fc32056e8ae171b81e0bac222f21b0
- vecnormalize_sha256: 37a9cf28651731ba90593ce155bec5c717d113186a5a1fbe175978ab407b6033

## 3. Training route

- Pure PPO from scratch was unstable.
- Reward hacking was identified and fixed before final policy selection.
- BC v1 exposed closed-loop covariate shift.
- DAgger-lite plus BC v2 succeeded as a stronger initialization route.
- Conservative PPO fine-tuning from BC v2 succeeded, and ppo_step_2500 was selected by deterministic sweep.

## 4. World0 validation

- Phase 7.1H world0 robustness: 30/30 success, success_rate 1.0000.
- target_visible_ratio_mean: 1.0000
- final_distance_mean: 0.7836 m
- min_distance_mean: 0.7834 m
- RAW_TIMEOUT count: 0
- OFFBOARD drop count: 0
- reset_pollution_detected: false

World0 has passed for the frozen policy.

## 5. World1 sparse validation

- Phase 7.2A world1 sparse zero-shot: 30/30 success, success_rate 1.0000.
- Phase 7.2B world1 sparse robustness/stress: 90/90 success, success_rate 1.0000.
- collision/emergency failure rate: 0.0000
- out_of_bounds rate: 0.0000
- height_violation rate: 0.0000
- OFFBOARD drop count: 0
- front-obstacle safety intervention: passed

World1 sparse obstacle has passed for the frozen policy.

## 6. Woods_easy validation

- Phase 7.3A woods_easy zero-shot: 29/30 success, success_rate 0.9667.
- Phase 7.3B woods_easy robustness/stress: 90/90 success, success_rate 1.0000.
- Phase 7.3C woods_easy reset fix: 50/50 reset success, reset_success_rate 1.0000.
- robustness/stress target_visible_ratio_mean: 0.9987
- robustness/stress collision/emergency failure rate: 0.0000
- reset_pollution_detected: false

Woods_easy has passed for the frozen policy.

## 7. Random_woods validation

- Phase 7.4A random_woods zero-shot: 29/30 success, success_rate 0.9667.
- Phase 7.4A timeout rate: 0.0333
- Phase 7.4A target_visible_ratio_mean: 1.0000
- Phase 7.4A final_distance_mean: 0.7870 m
- Phase 7.4A min_distance_mean: 0.7867 m
- Phase 7.4B random_woods robustness/stress: 88/90 success, success_rate 0.9778.
- Phase 7.4B Group A success rate: 0.9667
- Phase 7.4B Group B success rate: 1.0000
- Phase 7.4B Group C success rate: 0.9500
- Phase 7.4B timeout rate: 0.0222
- Phase 7.4B collision/emergency failure rate: 0.0111
- Phase 7.4B target_visible_ratio_mean: 0.9918
- Phase 7.4B final_distance_mean: 0.8004 m
- Phase 7.4B min_distance_mean: 0.7977 m
- Phase 7.4B RAW_TIMEOUT count: 0
- Phase 7.4B OFFBOARD drop count: 0
- Phase 7.4B reset_pollution_detected: false
- main failure modes: emergency_failure=1; timeout=2

Random_woods zero-shot and robustness/stress have passed for the frozen policy. The polluted first robustness attempt at `attempt_reset_polluted_20260521_013011` is excluded from this final registry.

## 8. Safety intervention validation

- world1 sparse front-obstacle safety intervention: passed.
- woods_easy Group D safety intervention: passed.
- random_woods Group D safety intervention: passed.
- random_woods Group D result: danger_seen=yes, safety_triggered=yes, filtered_vx_body<=0=yes, no_collision=yes, no_offboard=yes, recovered_or_safe=yes.

Safety-filter closed-loop validation is complete for the simulated gates above. This does not validate real-flight safety behavior.

## 9. Final artifacts

- policy_manifest.json
- final_results_table.csv
- final_results_table.md
- reproduce_commands.md
- final_phase7_world0_world1_report.md
- final_phase7_world0_world1_woods_easy_report.md
- final_phase7_world0_world1_woods_easy_random_woods_report.md
- phase7_4c_registry_update_report.md

## 10. Reproduction commands

Use `/home/whk/vf_ws/outputs/final_policy_registry/reproduce_commands.md` for the current evaluation-only launch and rosrun commands. The commands load the frozen registered model and frozen VecNormalize artifact from final_policy_registry.

## 11. Limitations

Allowed claims:

- world0 has passed.
- world1 sparse obstacle has passed.
- woods_easy has passed.
- random_woods zero-shot has passed.
- random_woods robustness/stress has passed.
- The current frozen policy has completed multi-scenario simulation validation for world0, world1 sparse, woods_easy, and random_woods.

Disallowed claims:

- all woods have passed.
- dense woods has passed.
- woods_hard has passed.
- dynamic obstacles have passed.
- real hardware deployment has passed.
- real RGB-D perception has passed.

## 12. Next phase recommendation

Proceed to a separate dense woods or woods_hard gate if the goal is broader woods coverage. Do not claim all woods until dense woods and woods_hard are independently validated. Dynamic obstacles, real RGB-D, and real flight require separate validation phases.
