"""Train a separate research method or one of its matched baselines."""

import hydra
from omegaconf import DictConfig

from learning.config import execute_workflow
from learning.supermimic.trainer import train, validate_config


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_supermimic"
)
def main(config: DictConfig):
    from learning.config.runtime import resolved_mapping

    validate_config(resolved_mapping(config))
    execute_workflow(config, train)


if __name__ == "__main__":
    main()
