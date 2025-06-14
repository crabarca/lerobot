
import logging
import draccus
import yaml

from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging
from lerobot.record import RecordConfig

# Select your device
device = "mps"

smolvla_base = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(smolvla_base)
policy.reset()

# # Print policy features
print(policy.config.input_features)

def read_config_from_yaml(file_path: str):
  with open(file_path, 'r') as file:
    data = yaml.safe_load(file)
    config = RecordConfig(**data)
    assert isinstance(config, RecordConfig), "Config must be an instance of RecordConfig"
    return config

@draccus.wrap()
def interact(cfg: RecordConfig):
    init_logging()
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    obs = robot.get_observation()
    logging.info("Observation:", obs)

    while True:
      obs = robot.get_observation()
      observation_frame = build_dataset_frame(cfg.dataset.features, obs, prefix="observation")
      action_values = predict_action(
         observation_frame,
         policy,
         'mps',
         True,
         task=cfg.dataset.task,
         robot=robot.robot_type,
      )
      action = { key: action_values[i].item() for i, key in enumerate(robot.action_features) }

      # Create the policy input dictionary
      sent_action = robot.send_action(action)
      logging.info(sent_action)
      busy_wait(1)  # Adjust the sleep time as needed
    

if __name__ == "__main__":
    interact()

