#!/usr/bin/env python3
"""
Test script to verify that the record command works with the fixes.
This script will test the record functionality with minimal parameters.
"""

import sys
import os
import time

import numpy as np

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.common.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.common.utils.control_utils import sanity_check_dataset_name
from lerobot.replay import DatasetReplayConfig, ReplayConfig, replay

# Add the lerobot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot'))

from lerobot.common.cameras import opencv
from lerobot.common.cameras.configs import CameraConfig
from lerobot.record import record
from lerobot.record import DatasetRecordConfig, RecordConfig


    
resting_position = { 
  'shoulder_pan.pos': -8.985507246376812, 
  'shoulder_lift.pos': -99.5085995085995, 
  'elbow_flex.pos': 98.93095768374164, 
  'wrist_flex.pos': 76.41395908543922, 
  'wrist_roll.pos': -5.103785103785114, 
  'gripper.pos': 3.20890635232482
}

HF_USER = 'crabarca'




def is_resting(robot, resting_position=resting_position):
    """Test the record functionality with minimal parameters."""

    position = robot.get_observation()

    obs_pos = [position[key] for key in resting_position.keys()]
    rest_pos = [resting_position[key] for key in resting_position.keys()]

    # Calculate the euclidean distance
    euclidean_distance = np.linalg.norm(np.array(obs_pos) - np.array(rest_pos))

    print(f"Observed position: {obs_pos}")
    print(f"Resting position: {rest_pos}")
    print(f"Euclidean distance: {euclidean_distance}")

    if (euclidean_distance < 10): 
        print("Resting position detected")
        return True
    else:
        print("Resting position not detected")
        return False

if __name__ == "__main__":
  pretrained_path = "vectorcrumb/trash_rover_v2"
  policy = SmolVLAPolicy.from_pretrained(pretrained_path)
  policy.config.device = "mps"
  policy.to(device="mps")

    # Create minimal configuration for testing
  robot_config = SO100FollowerConfig(
      port="/dev/tty.usbmodem59700725841",
      id="zack_fol",
      cameras={
          "up": OpenCVCameraConfig(
            index_or_path=0,
            width=1280,
            height=720,
            fps=30
          )
      }
  )

  dataset_replay = DatasetReplayConfig(
    repo_id=f"{HF_USER}/drop_paper_ball",
    episode=1
  )

  replay_config = ReplayConfig(
    robot=robot_config,
    dataset=dataset_replay,
    play_sounds=True
  )

  robot = make_robot_from_config(robot_config)

  action_features = hw_to_dataset_features(robot.action_features, "action", dataset_config.video)
  obs_features = hw_to_dataset_features(robot.observation_features, "observation", dataset_config.video)
  dataset_features = {**action_features, **obs_features}

  sanity_check_dataset_name(dataset_config.repo_id, policy)

  robot.connect()

  resting_position_detected = False

  while not resting_position_detected:
    success = is_resting(robot)
    # this will be detected while doing inference
    robot.send_action(resting_position) 
    resting_position_detected = True

  replay(replay_config)

  robot.disconnect()

