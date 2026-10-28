"""Train the canonical recursive DRLR/RMA phase-one recipe; dry-run by default."""

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow, register_configs
from learning.reinforcement.recursive import train_recursive

register_configs()


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_recursive_drlr_rma"
)
def main(config: DictConfig):
    execute_workflow(config, train_recursive)


if __name__ == "__main__":
    main()
