#!/usr/bin/env python3

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import yaml

from lerobot.configs import parser
from lerobot.record import RecordConfig


@dataclass
class Connect4Config:
  scenarios: list[RecordConfig]


def read_scenarios_from_yaml(file_path: str) -> Connect4Config:
  with open(file_path, 'r') as file:
    data = yaml.safe_load(file)
    scenarios = [RecordConfig(**scenario) for scenario in data.get('scenarios', [])]
    return Connect4Config(scenarios=scenarios)


@parser.wrap()
def record_moves(cfg: Connect4Config):
  logging.info(pformat(asdict(cfg)))
  print("Available scenarios:")
  for idx, scenario in enumerate(cfg.scenarios):
    print(f"{idx + 1}: {scenario.dataset.single_task}")

  selected = None
  while selected is None:
    try:
      choice = int(input("Select a scenario to record (enter number): "))
      if 1 <= choice <= len(cfg.scenarios):
        selected = cfg.scenarios[choice - 1]
      else:
        print("Invalid selection. Try again.")
    except ValueError:
      print("Please enter a valid number.")

  print(f"Starting recording for scenario: {selected.dataset.single_task}")
  # TODO: test this
  # record(cfg.scenario)


if __name__ == "__main__":
  record_moves()