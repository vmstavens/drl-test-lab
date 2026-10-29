"""Optional real-asset CPU checks; never download meshes in the test suite."""

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest

from envs.supermimic_hand import SuperMimicHand

ASSETS = Path(__file__).resolve().parents[1] / "data/assets/mujoco_menagerie"


@pytest.mark.skipif(
    not (ASSETS / "leap_hand/assets").is_dir(),
    reason="Fetch pinned LEAP assets explicitly to run the hand integration checks",
)
@pytest.mark.parametrize("mode", ["reorient", "rotate"])
def test_real_hand_physics_fixed_goals_and_executed_increment(mode):
    environment = SuperMimicHand(
        config_overrides=dict(
            impl="jax",
            menagerie_path=str(ASSETS),
            mode=mode,
        )
    )
    try:
        initial = jax.jit(environment.reset)(jax.random.PRNGKey(0))
        following = jax.jit(environment.step)(initial, jp.full(16, 10.0))
        executed = np.asarray(following.info["executed_action"])
        assert np.max(np.abs(executed)) <= 0.5 + 1e-6
        np.testing.assert_allclose(
            executed, np.asarray(following.data.ctrl - initial.data.ctrl)
        )
        realized = environment.model_action(
            environment.model_state(initial), jp.ones(16)
        )
        np.testing.assert_allclose(realized, executed / 0.5, atol=1e-6)
        np.testing.assert_array_equal(following.info["goal"], initial.info["goal"])
        np.testing.assert_allclose(
            np.linalg.norm(following.data.qpos[19:23]), 1.0, atol=1e-5
        )
        assert np.isfinite(following.obs).all()
        assert np.isfinite(float(following.reward))
        assert environment.model_state(following).shape == (61,)
    finally:
        environment.close()
