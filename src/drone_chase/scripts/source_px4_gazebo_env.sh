#!/usr/bin/env bash
# Source this before launching PX4/Gazebo based drone_chase phases.
# It keeps the custom chase models and PX4 SITL Gazebo plugins on the same path.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

export DRONE_CHASE_WS="${DRONE_CHASE_WS:-/home/whk/vf_ws}"
export PX4_ROOT="${PX4_ROOT:-/home/whk/PX4-Autopilot}"
export PX4_BUILD_DIR="${PX4_BUILD_DIR:-${PX4_ROOT}/build/px4_sitl_default}"
export PX4_SITL_GAZEBO="${PX4_ROOT}/Tools/sitl_gazebo"

source /opt/ros/noetic/setup.bash
source "${DRONE_CHASE_WS}/devel/setup.bash"
source "${PX4_ROOT}/Tools/setup_gazebo.bash" "${PX4_ROOT}" "${PX4_BUILD_DIR}"

append_path_once() {
  local var_name="$1"
  local path_entry="$2"
  if [[ -z "${path_entry}" ]]; then
    return
  fi
  local current_value="${!var_name:-}"
  case ":${current_value}:" in
    *":${path_entry}:"*) ;;
    *)
      if [[ -z "${current_value}" ]]; then
        export "${var_name}=${path_entry}"
      else
        export "${var_name}=${current_value}:${path_entry}"
      fi
      ;;
  esac
}

append_path_once ROS_PACKAGE_PATH "${PX4_ROOT}"
append_path_once ROS_PACKAGE_PATH "${PX4_SITL_GAZEBO}"
append_path_once GAZEBO_MODEL_PATH "${DRONE_CHASE_WS}/src/drone_chase/models"
append_path_once GAZEBO_MODEL_PATH "${PX4_SITL_GAZEBO}/models"
append_path_once GAZEBO_PLUGIN_PATH "${PX4_SITL_GAZEBO}/build"
append_path_once LD_LIBRARY_PATH "${PX4_SITL_GAZEBO}/build"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export ROS_HOME="${ROS_HOME:-${DRONE_CHASE_WS}/.ros}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${DRONE_CHASE_WS}/log}"
