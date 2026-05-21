# Phase 7.3C Registry Update Report

1. 是否更新 policy_manifest.json：是。
2. 是否更新 final_results_table：是，已更新 final_results_table.csv 与 final_results_table.md。
3. 是否更新 reproduce_commands.md：是。
4. 是否生成 final_phase7_world0_world1_woods_easy_report.md：是。
5. world0 success rate：1.0000 (30/30)。
6. world1 zero-shot success rate：1.0000 (30/30)。
7. world1 robustness success rate：1.0000 (90/90)。
8. woods_easy zero-shot success rate：0.9667 (29/30)。
9. woods_easy robustness success rate：1.0000 (90/90)。
10. woods_easy reset success rate：1.0000 (50/50)。
11. 当前 frozen policy 路径：/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip。
12. 当前验证边界：world0、world1 sparse obstacle、woods_easy zero-shot、woods_easy robustness/stress、woods_easy reset gate 已通过；random_woods、dense woods、woods_hard、dynamic obstacles、真机部署、真实 RGB-D 感知未验证。
13. 是否允许进入 random_woods zero-shot：是，作为新的独立验证阶段。
14. 是否允许声称全部 woods 已通过：否。

Notes:

- Phase 7.3C only updates registry artifacts with completed woods_easy results.
- No policy training, fine-tuning, reward change, safety-filter control change, action-mapping change, or policy artifact replacement was performed.
