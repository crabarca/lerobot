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
from lerobot.record import RecordConfig, record
from lerobot.record import DatasetRecordConfig
import rerun as rr
import numpy as np

import os

shutil.rmtree("/Users/alvarez/.cache/huggingface/lerobot/vectorcrumb/eval_trash_pickup_v1", ignore_errors=True)


# robot_config = SO101FollowerConfig(
#     port="/dev/tty.usbmodem59700733641",
#     id="fol",
#     cameras={
#         "gripper": OpenCVCameraConfig(
#             index_or_path=0,
#             width=1280,
#             height=720,
#             fps=30
#         ),
#         "front": OpenCVCameraConfig(
#             index_or_path=1,
#             width=1280,
#             height=720,
#             fps=30
#         )
#     }
# )
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

robot = SO100Follower(robot_config)

robot.connect()

fps=20
display_data = True
dataset_config=DatasetRecordConfig(
    repo_id="crabarca/eval_trash_rover_pickup_v1",
    single_task="Grab the brown paper ball from the platform",
    episode_time_s=120,
    num_episodes=2,
    push_to_hub=False,
    fps=20
)

pretrained_path = "armerprinz/smolvla-tp-v1"
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
    robot_type="SO101Follower",
    features=dataset_features,
    use_videos=dataset_config.video,
    image_writer_processes=dataset_config.num_image_writer_processes,
    image_writer_threads=dataset_config.num_image_writer_threads_per_camera * len(robot.cameras),
)
prompt_queue = queue.Queue()
prompt_queue.put("Bottle caps and paper balls are trash. Put them in the correct box.")
task = prompt_queue.get()
def get_prompt_thread():
    while True:
        new_prompt = input("\nPrompt the robot: ")
        prompt_queue.put(new_prompt)
        time.sleep(0.1)

_init_rerun(session_name="inference")


input_thread = threading.Thread(target=get_prompt_thread, daemon=True)
input_thread.start()

while True:
    try:
        start_time = time.perf_counter()
        observation = robot.get_observation()
        # task = "Grab the blue bottle cap and put it into the left box"
        # if not prompt_queue.empty():
        #     print("\n\n\n")
        #     task = prompt_queue.get()
        #     print("Got a new task:", task)
        #     if task == "end":
        #         break
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
        is_resting()


        if display_data:
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalars(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
                elif isinstance(val, str):
                    rr.log(f"observation.{obs}", rr.TextLog(f"Task is: {val}", level=rr.TextLogLevel.INFO))
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalars(val))
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
    except KeyboardInterrupt:
        break

robot.disconnect()
# record_config = RecordConfig(
#     robot=robot_config,
#     dataset=dataset_config,
#     policy=PreTrainedConfig.from_pretrained(pretrained_path)
# )
# record_config.policy.pretrained_path=pretrained_path

# record(record_config)