# Final Results Summary

Final frozen policy:

- model: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_policy.zip`
- vecnormalize: `/home/whk/vf_ws/outputs/final_policy_registry/best_world0_world1_vecnormalize.pkl`

| validation gate | world | episodes | success | success_rate | notes |
| --- | --- | --- | --- | --- | --- |
| world0 robustness | world_0 | 30 | 30/30 | 1.0000 | Passed |
| world1 zero-shot | world_1 sparse | 30 | 30/30 | 1.0000 | Passed |
| world1 robustness | world_1 sparse | 90 | 90/90 | 1.0000 | Passed |
| woods_easy zero-shot | woods_easy | 30 | 29/30 | 0.9667 | Passed |
| woods_easy robustness | woods_easy | 90 | 90/90 | 1.0000 | Passed |
| random_woods zero-shot | random_woods | 30 | 29/30 | 0.9667 | Passed |
| random_woods robustness | random_woods | 90 | 88/90 | 0.9778 | Passed; safety intervention passed |

Boundary:

- Validated: world0, world1 sparse obstacle, woods_easy, random_woods.
- Not validated: dense woods, woods_hard, dynamic obstacles, real RGB-D, real flight.
