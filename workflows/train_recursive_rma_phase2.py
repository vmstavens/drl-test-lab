"""Train the history adapter of an accepted actor; dry-run by default."""

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.reinforcement.rma import train_phase2

register_configs()


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_recursive_rma_phase2",
)
def main(config: DictConfig):
    execute_workflow(config, train_phase2)


if __name__ == "__main__":
    main()
