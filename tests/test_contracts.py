"""Lightweight source/default and semantic contract checks for core CI."""

import json
from pathlib import Path

import pytest

from learning.contracts import LearningContract
from learning.reinforcement.budget import resolve_training_budget
from tests.test_learning_infrastructure import compose_workflow


def small_config():
    return compose_workflow("train_recursive_drlr_rma")


def test_default_recipe_matches_saved_source_for_retained_learning_settings():
    config = compose_workflow("train_recursive_drlr_rma")
    source = json.loads(
        (
            Path(__file__).parents[1] / "docs/canonical-learning-provenance.json"
        ).read_text()
    )["source_rl"]
    for name in (
        "num_envs",
        "memory_size",
        "batch_size",
        "gradient_steps",
        "warmup_vector_steps",
        "discount_factor",
        "polyak",
        "actor_learning_rate",
        "critic_learning_rate",
        "actor_hidden_sizes",
        "critic_hidden_sizes",
        "bellman_target_min",
        "bellman_target_max",
        "initial_action_std",
        "minimum_action_std",
        "maximum_action_std",
        "reference_replay_fraction",
        "critic_layer_norm",
        "freeze_entropy_temperature",
        "initial_entropy_temperature",
    ):
        assert config["rl"][name] == source[name], name
    assert config["runtime"]["dry_run"]
    assert config["rl"]["previous_action_source"] == "executed"


def test_contract_rejects_equal_dimensions_with_different_frames_or_timing():
    contract = LearningContract.from_config(small_config())
    for key, value in (
        ("coordinate_frame", "robot_base"),
        ("control_dt", 0.004),
        ("action_units", ["m", "m"]),
        ("preprocessing", "filtered_v2"),
    ):
        recorded = contract.to_dict()
        recorded[key] = value
        with pytest.raises(ValueError, match="contract differs"):
            contract.require(recorded)


def test_budget_accounts_for_parallel_environments():
    budget = resolve_training_budget(
        {"budget": {"value": 1001, "unit": "transitions"}}, num_envs=128
    )
    assert budget.vector_steps == 8 and budget.transitions == 1024
