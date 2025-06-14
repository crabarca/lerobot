#!/usr/bin/env python3
"""
Test script to verify that the record command works with the fixes.
This script will test the record functionality with minimal parameters.
"""

import sys
import os

# Add the lerobot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot'))

from lerobot.common.cameras import opencv
from lerobot.common.cameras.configs import CameraConfig
from lerobot.record import record
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.record import DatasetRecordConfig, RecordConfig


HF_USER = "crabarca"

def test_record_fix():
    """Test the record functionality with minimal parameters."""
    
    # Create minimal configuration for testing
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem59700725841",
        id="zack_follower",
        cameras={
            "gripper": CameraConfig(
              type=opencv,
              index_or_path=0,
              width=1280,
              height=720,
              fps=30
            ),
            "top": CameraConfig(
              type=opencv,
              index_or_path=1,
              width=1920,
              height=1080,
              fps=60
            )
        }
    )
    
    teleop_config = SO100LeaderConfig(
        port="/dev/tty.usbmodem5970073539",
        id="zack_leader"
    )
    
    dataset_config = DatasetRecordConfig(
        repo_id=f"{HF_USER}/dataset_test6",
        single_task="Grab the blue bottle cap",
        num_episodes=1,
        episode_time_s=5,  # Short episode for testing
        reset_time_s=2,
        push_to_hub=False,  # Don't push to hub for testing
        fps=20  # Lower fps for testing
    )
    
    record_config = RecordConfig(
        robot=robot_config,
        dataset=dataset_config,
        teleop=teleop_config,
        display_data=False,
        play_sounds=False
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