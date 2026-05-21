# Final Phase 7 Results Table

| stage | world | policy | episodes | success_rate | timeout_rate | collision_rate | out_of_bounds_rate | height_violation_rate | target_visible_ratio | final_distance_mean | min_distance_mean | raw_timeout_count | emergency_count | depth_stop_count | offboard_drop_count | reset_pollution | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 7.1H world0 robustness | world_0 | ppo_step_2500 | 30 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7836 | 0.7834 | 0 | 23 | 0 | 0 | false | World0 robustness gate passed; 30/30 success. |
| Phase 7.2A world1 zero-shot | world_1_sparse | ppo_step_2500 | 30 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7842 | 0.7842 | 0 | 18 | 0 | 0 | false | World1 sparse zero-shot gate passed; 30/30 success. |
| Phase 7.2B world1 robustness/stress | world_1_sparse | ppo_step_2500 | 90 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9892 | 0.7861 | 0.7861 | 0 | 153 | 0 | 0 | false | World1 sparse robustness/stress passed; groups A/B/C/D passed. |
| Phase 7.3A woods_easy zero-shot | woods_easy | ppo_step_2500 | 30 | 0.9667 | 0.0000 | 0.0333 | 0.0000 | 0.0000 | 0.9973 | 0.8083 | 0.8083 | 0 | 19 | 35 | 0 | false | Woods_easy zero-shot passed; 29/30 success. |
| Phase 7.3B woods_easy robustness/stress | woods_easy | ppo_step_2500 | 90 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9987 | 0.7846 | 0.7846 | 0 | 77 | 0 | 0 | false | Woods_easy robustness/stress passed; groups A/B/C/D passed. |
| Phase 7.3C woods_easy reset fix | woods_easy | reset_gate | 50 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |  | 0 | 0 | 0 | 0 | false | Woods_easy reset gate passed; 50/50 reset success. |
| Phase 7.4A random_woods zero-shot | random_woods | ppo_step_2500 | 30 | 0.9667 | 0.0333 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7870 | 0.7867 | 0 | 4 | 0 | 0 | false | Random_woods zero-shot passed; 29/30 success. |
| Phase 7.4B random_woods robustness/stress | random_woods | ppo_step_2500 | 90 | 0.9778 | 0.0222 | 0.0111 | 0.0000 | 0.0000 | 0.9918 | 0.8004 | 0.7977 | 0 | 205 | 50 | 0 | false | Random_woods robustness/stress passed; groups A/B/C passed and group D safety intervention passed; polluted attempt excluded. |
