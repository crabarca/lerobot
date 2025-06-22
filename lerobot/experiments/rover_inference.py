import pathlib
import queue
import shutil
import threading
import time

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.so100_follower.so100_follower import SO100Follower
from lerobot.common.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.common.robots.so101_follower.so101_follower import SO101Follower
from lerobot.common.utils.control_utils import predict_action, sanity_check_dataset_name
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs.policies import PreTrainedConfig
from lerobot.experiments.wave import WaveRoverControl
from lerobot.record import RecordConfig, record
from lerobot.record import DatasetRecordConfig
import rerun as rr
import numpy as np

import os

from lerobot.replay import DatasetReplayConfig, ReplayConfig, replay


HF_USER = "crabarca"
shutil.rmtree(f"/Users/cristobal/.cache/huggingface/lerobot/{HF_USER}/eval_rover_pickup_v1", ignore_errors=True)
    
resting_position = { 
  'shoulder_pan.pos': -8.985507246376812, 
  'shoulder_lift.pos': -99.5085995085995, 
  'elbow_flex.pos': 98.93095768374164, 
  'wrist_flex.pos': 76.41395908543922, 
  'wrist_roll.pos': -5.103785103785114, 
  'gripper.pos': 3.20890635232482
}

def is_resting(robot, resting_position=resting_position, threshold=50):
    """Test the record functionality with minimal parameters."""

    position = robot.get_observation()

    obs_pos = [position[key] for key in resting_position.keys()]
    rest_pos = [resting_position[key] for key in resting_position.keys()]

    # Calculate the euclidean distance
    euclidean_distance = np.linalg.norm(np.array(obs_pos) - np.array(rest_pos))

    print(f"Observed position: {obs_pos}")
    print(f"Resting position: {rest_pos}")
    print(f"Euclidean distance: {euclidean_distance} - Threshold: {threshold}" )

    if (euclidean_distance < threshold): 
        print("Resting position detected")
        return True
    else:
        print("Resting position not detected")
        return False

# ROBOT CONFIGS
# TODO: Add configs
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem59700725841",
    id="zack_fol",
    cameras={
        "up": OpenCVCameraConfig(
          index_or_path=1,
          width=1280,
          height=720,
          fps=30
        )
    }
)


robot = SO100Follower(robot_config)
robot.connect()
print('connect to robot')

fps=20
display_data = True
dataset_config=DatasetRecordConfig(
    repo_id=f"{HF_USER}/eval_rover_pickup_v1",
    single_task="Grab the brown paper ball from the platform and keep it",
    episode_time_s=120,
    num_episodes=2,
    push_to_hub=False,
    fps=20
)

pretrained_path = "vectorcrumb/trash_rover_v2"
policy = SmolVLAPolicy.from_pretrained(pretrained_path)
policy.config.device = "mps"
policy.to(device="mps")

action_features = hw_to_dataset_features(robot.action_features, "action", dataset_config.video)
obs_features = hw_to_dataset_features(robot.observation_features, "observation", dataset_config.video)
dataset_features = {**action_features, **obs_features}

sanity_check_dataset_name(dataset_config.repo_id, policy)
dataset = LeRobotDataset.create(
    dataset_config.repo_id,
    dataset_config.fps,
    root=dataset_config.root,
    robot_type="SO100Follower",
    features=dataset_features,
    use_videos=dataset_config.video,
    image_writer_processes=dataset_config.num_image_writer_processes,
    image_writer_threads=dataset_config.num_image_writer_threads_per_camera * len(robot.cameras),
)
prompt_queue = queue.Queue()
prompt_queue.put("Grab the brown paper ball from the platform and keep it")
task = prompt_queue.get()
def get_prompt_thread():
    while True:
        new_prompt = input("\nPrompt the robot: ")
        prompt_queue.put(new_prompt)
        time.sleep(0.1)

_init_rerun(session_name="inference")

reached_resting = False
actions = 0
beginning_time = time.time()
while not reached_resting:
    try:
        start_time = time.perf_counter()
        observation = robot.get_observation()
        observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
        action_values = predict_action(
            observation_frame,
            policy,
            get_safe_torch_device(policy.config.device),
            policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type
        )
        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        sent_action = robot.send_action(action)
        actions += 1

        if display_data:
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalars(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
                elif isinstance(val, str):
                    rr.log(f"observation.{obs}", rr.TextLog(f"Task is: {val}", level=rr.TextLogLevel.INFO), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalars(val))

        if is_resting(robot, resting_position, threshold=20) and actions > 5 and time.time() - beginning_time > 8:
            print('reached resting')
            reached_resting = True 
            break

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
        print('waiting')


    except KeyboardInterrupt:
        break

# move rover

# rover = WaveRoverControl("192.168.8.127")
# time.sleep(0.1)
# rover.set_speed(-0.2, -0.2)
# time.sleep(3)
# rover.emergency_stop()
# time.sleep(25)
input("press enter to drop ball")

print('droping ball')
# replay episode
dataset_replay = DatasetReplayConfig(
  repo_id=f"{HF_USER}/drop_paper_ball",
  episode=1
)

replay_config = ReplayConfig(
  robot=robot_config,
  dataset=dataset_replay,
  play_sounds=True
)

replay(replay_config)

# rover.set_speed(0.2, 0.2)
# time.sleep(2)
# rover.emergency_stop()


