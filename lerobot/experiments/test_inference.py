#!/usr/bin/env python3
"""
Test script to verify that the record command works with the fixes.
This script will test the record functionality with minimal parameters.
"""

import sys
import os

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.common.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.configs.policies import PreTrainedConfig

# Add the lerobot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot'))

from lerobot.common.cameras import opencv
from lerobot.common.cameras.configs import CameraConfig
from lerobot.record import record
from lerobot.record import DatasetRecordConfig, RecordConfig


HF_USER = "crabarca"

def test_record_fix():
    """Test the record functionality with minimal parameters."""
    
    # Create minimal configuration for testing
    robot_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem59700733641",
        id="fol_zack",
        cameras={
            "gripper": OpenCVCameraConfig(
              index_or_path=1,
              width=1280,
              height=720,
              fps=30
            ),
            "front": OpenCVCameraConfig(
              index_or_path=2,
              width=1280,
              height=720,
              fps=30
            )
        }
    )
    
    # teleop_config = SO101LeaderConfig(
    #     port="/dev/tty.usbmodem59700735391",
    #     id="lea_zack"
    # )
    
    dataset_config = DatasetRecordConfig(
        repo_id=f"{HF_USER}/trash_pickup_v2",
        # single_task="Grab the blue bottle cap and put it into the left box",
        single_task="Grab the blue bottle cap and put it into the left box",
        num_episodes=2,
        episode_time_s=50,  # Short episode for testing
        reset_time_s=5,
        push_to_hub=True,  # Don't push to hub for testing
        fps=20  # Lower fps for testinGrab the brown paper ball from the platform and keep itGrab the brown paper ball from the platform and keep itg
    )

    # pre_trained_config = PreTrainedConfig(
    #     path="armerprinz/smolvla-tp-v2",
    #     device="mps"
    # )
    pre_trained_config = SmolVLAConfig(
        path="armerprinz/smolvla-tp-v2",
        device="mps"
    )
    
    record_config = RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        # teleop=teleop_config,
        display_data=True,
        play_sounds=True,
        resume=True,
        policy=pre_trained_config
    )
    
    try:
        print("Starting record test...")
        dataset = record(record_config)
        print(f"Record test completed successfully! Dataset: {dataset}")
        return True
    except Exception as e:
        print(f"Record test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_record_fix()
    sys.exit(0 if success else 1) 