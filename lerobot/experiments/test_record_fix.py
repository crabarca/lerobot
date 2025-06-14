#!/usr/bin/env python3
"""
Test script to verify that the record command works with the fixes.
This script will test the record functionality with minimal parameters.
"""

import sys
import os

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig

# Add the lerobot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot'))

from lerobot.common.cameras import opencv
from lerobot.common.cameras.configs import CameraConfig
from lerobot.record import record
from lerobot.record import DatasetRecordConfig, RecordConfig


HF_USER = "vectorcrumb"

def test_record_fix():
    """Test the record functionality with minimal parameters."""
    
    # Create minimal configuration for testing
    robot_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem59700733641",
        id="fol",
        cameras={
            "gripper": OpenCVCameraConfig(
              index_or_path=0,
              width=1280,
              height=720,
              fps=30
            ),
            "front": OpenCVCameraConfig(
              index_or_path=1,
              width=1280,
              height=720,
              fps=30
            )
        }
    )
    
    teleop_config = SO101LeaderConfig(
        port="/dev/tty.usbmodem59700740331",
        id="lea"
    )
    
    dataset_config = DatasetRecordConfig(
        repo_id=f"{HF_USER}/trash_pickup_v1",
        # single_task="Grab the blue bottle cap and put it into the left box",
        single_task="Grab the brown paper ball and put it into the right box",
        num_episodes=30,
        episode_time_s=15,  # Short episode for testing
        reset_time_s=5,
        push_to_hub=True,  # Don't push to hub for testing
        fps=20  # Lower fps for testing
    )
    
    record_config = RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        teleop=teleop_config,
        display_data=True,
        play_sounds=True,
        resume=True
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