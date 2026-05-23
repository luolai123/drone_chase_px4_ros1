cmake_minimum_required(VERSION 3.0.2)
project(drone_chase)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  gazebo_msgs
  gazebo_ros
  geometry_msgs
  mavros_msgs
  mavros
  sensor_msgs
  std_msgs
  std_srvs
  cv_bridge
  image_transport
  message_filters
)

add_message_files(
  FILES
  TargetState.msg
  DepthRisk.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  message_runtime
  gazebo_msgs
  gazebo_ros
  geometry_msgs
  mavros_msgs
  mavros
  sensor_msgs
  std_msgs
  std_srvs
  cv_bridge
  image_transport
  message_filters
)

# Install scripts
catkin_install_python(PROGRAMS
  scripts/offboard_velocity_test.py
  scripts/spawn_red_ball.py
  scripts/spawn_random_obstacles.py
  scripts/spawn_random_woods.py
  scripts/reset_chase_world.py
  scripts/reset_random_woods_world.py
  scripts/red_ball_detector.py
  scripts/depth_risk_estimator.py
  scripts/diagnostic_hover_1p5m.py
  scripts/diagnostic_spawn_front_obstacle.py
  scripts/rule_based_chaser.py
  scripts/cmd_vel_forwarder.py
  scripts/safety_filter_node.py
  scripts/phase6_env_smoke_test.py
  scripts/phase6_random_policy_rollout.py
  scripts/phase6_random_env_test.py
  scripts/phase6_rule_env_test.py
  scripts/phase6_episode_reset.py
  scripts/phase7_check_training_env.py
  scripts/phase7_expert_env_sanity.py
  scripts/phase7_action_direction_audit.py
  scripts/phase7_reward_audit.py
  scripts/phase7_python_preflight.py
  scripts/phase7_reset_stability_check.py
  scripts/phase7_train_ppo_lowdim.py
  scripts/phase7_train_ppo_from_bc.py
  scripts/phase7_eval_ppo_lowdim.py
  scripts/phase7_plot_training_csv.py
  scripts/phase7_checkpoint_sweep.py
  scripts/phase7_failed_rollout_audit.py
  scripts/phase7_command_pipeline_offline_audit.py
  scripts/phase7_best_command_trace_eval.py
  scripts/phase7_forward_action_intervention.py
  scripts/phase7_collect_expert_demos.py
  scripts/phase7_train_bc_policy.py
  scripts/phase7_eval_bc_policy.py
  scripts/phase7_bc_online_error_audit.py
  scripts/phase7_bc_action_ablation_eval.py
  scripts/phase7_collect_dagger_demos.py
  scripts/phase7_build_bc_aggregated_dataset.py
  scripts/phase7_world0_best_policy_robustness_eval.py
  scripts/phase7_world1_zeroshot_eval.py
  scripts/phase7_world1_robustness_eval.py
  scripts/phase7_build_final_results_table.py
  scripts/phase7_woods_easy_zeroshot_eval.py
  scripts/phase7_woods_easy_robustness_eval.py
  scripts/phase7_woods_easy_failure_diagnosis.py
  scripts/phase7_woods_easy_reset_fix_eval.py
  scripts/phase7_random_woods_zeroshot_eval.py
  scripts/phase7_random_woods_robustness_eval.py
  scripts/phase8_perception_diagnostic_recorder.py
  scripts/phase8_perception_feature_regression_eval.py
  scripts/phase8_policy_feature_ablation_eval.py
  scripts/phase8_perception_debug_correlation_analysis.py
  scripts/phase8_build_observation_v2_dataset.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY config envs models urdf worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
