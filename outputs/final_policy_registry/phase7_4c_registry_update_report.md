# Phase 7.4C Registry Update Report

1. 是否更新 policy_manifest.json：是。
2. 是否更新 final_results_table.csv：是。
3. 是否更新 final_results_table.md：是。
4. 是否更新 reproduce_commands.md：是。
5. 是否生成 final_phase7_world0_world1_woods_easy_random_woods_report.md：是。
6. world0 success rate：1.0000 (30/30)。
7. world1 zero-shot success rate：1.0000 (30/30)。
8. world1 robustness success rate：1.0000 (90/90)。
9. woods_easy zero-shot success rate：0.9667 (29/30)。
10. woods_easy robustness success rate：1.0000 (90/90)。
11. random_woods zero-shot success rate：0.9667 (29/30)。
12. random_woods robustness success rate：0.9778 (88/90)。
13. safety intervention 是否通过：是，random_woods Group D safety intervention passed。
14. 是否允许声称 world0 已通过：是。
15. 是否允许声称 world1 sparse 已通过：是。
16. 是否允许声称 woods_easy 已通过：是。
17. 是否允许声称 random_woods 已通过：是。
18. 是否允许声称全部 woods 已通过：否。
19. 当前 frozen policy 是否保持不变：是，model 与 vecnormalize 路径和 sha256 均保持不变。
20. 当前验证边界：已通过 world0、world1 sparse obstacle、woods_easy、random_woods；未验证 dense woods、woods_hard、dynamic obstacles、真机部署、真实 RGB-D 感知。
21. 下一阶段建议：进入 dense woods 或 woods_hard 独立 gate；不要用 random_woods 结果外推全部 woods。

Notes:

- Phase 7.4C only updates final_policy_registry reports, tables, manifest, and reproduction documentation.
- No training, fine-tuning, reward change, safety-filter control change, action-mapping change, Phase 1/2/3 logic change, model overwrite, or VecNormalize overwrite was performed.
- The polluted Phase 7.4B attempt archived at `/home/whk/vf_ws/outputs/phase7/random_woods_robustness_from_final_policy/attempt_reset_polluted_20260521_013011` is excluded from final registry metrics.
