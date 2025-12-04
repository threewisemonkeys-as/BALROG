import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from balrog.environments.minihack.minihack_env import make_minihack_env
from balrog.environments.nle.nle_env import make_nle_env


class MockConfig:
    """Mock config object for testing environment creation."""

    def __init__(self, include_lang_obs=True):
        self.agent = MockAgent()
        self.envs = MockEnvs(include_lang_obs)


class MockAgent:
    def __init__(self):
        self.max_image_history = 0


class MockEnvs:
    def __init__(self, include_lang_obs=True):
        self.minihack_kwargs = {
            "character": "@",
            "max_episode_steps": 100,
            "penalty_step": -0.01,
            "penalty_time": 0.0,
            "penalty_mode": "constant",
            "savedir": None,
            "save_ttyrec_every": 0,
            "skip_more": True,
            "include_lang_obs": include_lang_obs,
        }
        self.nle_kwargs = {
            "character": "@",
            "savedir": None,
            "save_ttyrec_every": 0,
            "skip_more": True,
            "include_lang_obs": include_lang_obs,
        }


@pytest.mark.parametrize("include_lang_obs", [True, False])
def test_minihack_language_obs_toggle(include_lang_obs):
    """Test that include_lang_obs toggle works correctly in MiniHack environments."""
    config = MockConfig(include_lang_obs=include_lang_obs)

    # Create environment with the toggle
    env = make_minihack_env(
        env_name="minihack",
        task="MiniHack-Room-5x5-v0",
        config=config,
        render_mode=None
    )

    # Reset environment to get initial observation
    obs, info = env.reset()

    # Check that the wrapper has the correct setting
    # Find the NLELanguageWrapper in the wrapper chain
    # Start with gym_env since GymV21CompatibilityV0 wraps the actual environment
    current_env = env.gym_env if hasattr(env, 'gym_env') else env
    nle_wrapper = None
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'NLELanguageWrapper':
            nle_wrapper = current_env
            break
        current_env = current_env.env

    assert nle_wrapper is not None, "NLELanguageWrapper not found in environment chain"
    assert nle_wrapper.include_lang_obs == include_lang_obs, \
        f"Expected include_lang_obs to be {include_lang_obs}, but got {nle_wrapper.include_lang_obs}"

    # Check that the observation content matches the toggle setting
    text_obs = obs.get("text", {})
    long_term_context = text_obs.get("long_term_context", "")

    # When include_lang_obs is True, "language observation" should be in the context
    # When False, it should not be present
    has_lang_obs = "language observation" in long_term_context

    assert has_lang_obs == include_lang_obs, \
        f"Expected language observation to be {'present' if include_lang_obs else 'absent'}, " \
        f"but it was {'present' if has_lang_obs else 'absent'}"

    # Verify that other observations are still present regardless of toggle
    assert "message" in long_term_context, "Message should always be present"
    assert "cursor" in long_term_context, "Cursor should always be present"

    env.close()


@pytest.mark.parametrize("include_lang_obs", [True, False])
def test_nle_language_obs_toggle(include_lang_obs):
    """Test that include_lang_obs toggle works correctly in NLE environments."""
    config = MockConfig(include_lang_obs=include_lang_obs)

    # Create NLE environment with the toggle
    env = make_nle_env(
        env_name="nle",
        task="NetHackChallenge-v0",
        config=config,
        render_mode=None
    )

    # Reset environment to get initial observation
    obs, info = env.reset()

    # Check that the wrapper has the correct setting
    # Start with gym_env since GymV21CompatibilityV0 wraps the actual environment
    current_env = env.gym_env if hasattr(env, 'gym_env') else env
    nle_wrapper = None
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'NLELanguageWrapper':
            nle_wrapper = current_env
            break
        current_env = current_env.env

    assert nle_wrapper is not None, "NLELanguageWrapper not found in environment chain"
    assert nle_wrapper.include_lang_obs == include_lang_obs, \
        f"Expected include_lang_obs to be {include_lang_obs}, but got {nle_wrapper.include_lang_obs}"

    # Check that the observation content matches the toggle setting
    text_obs = obs.get("text", {})
    long_term_context = text_obs.get("long_term_context", "")

    # When include_lang_obs is True, "language observation" should be in the context
    # When False, it should not be present
    has_lang_obs = "language observation" in long_term_context

    assert has_lang_obs == include_lang_obs, \
        f"Expected language observation to be {'present' if include_lang_obs else 'absent'}, " \
        f"but it was {'present' if has_lang_obs else 'absent'}"

    # Verify that other observations are still present regardless of toggle
    assert "message" in long_term_context, "Message should always be present"
    assert "cursor" in long_term_context, "Cursor should always be present"

    env.close()


def test_language_obs_default_value():
    """Test that include_lang_obs defaults to True when not specified."""
    # Create config without include_lang_obs specified
    config = MockConfig(include_lang_obs=True)
    config.envs.minihack_kwargs.pop("include_lang_obs")

    # Create environment
    env = make_minihack_env(
        env_name="minihack",
        task="MiniHack-Room-5x5-v0",
        config=config,
        render_mode=None
    )

    # Find the NLELanguageWrapper
    # Start with gym_env since GymV21CompatibilityV0 wraps the actual environment
    current_env = env.gym_env if hasattr(env, 'gym_env') else env
    nle_wrapper = None
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'NLELanguageWrapper':
            nle_wrapper = current_env
            break
        current_env = current_env.env

    assert nle_wrapper is not None, "NLELanguageWrapper not found"
    # Default should be True
    assert nle_wrapper.include_lang_obs == True, \
        "Expected include_lang_obs to default to True"

    env.close()


@pytest.mark.parametrize("vlm_mode", [True, False])
@pytest.mark.parametrize("include_lang_obs", [True, False])
def test_language_obs_with_vlm_modes(vlm_mode, include_lang_obs):
    """Test that include_lang_obs works correctly with both VLM and non-VLM modes."""
    config = MockConfig(include_lang_obs=include_lang_obs)

    # Set max_image_history to control VLM mode
    config.agent.max_image_history = 1 if vlm_mode else 0

    env = make_minihack_env(
        env_name="minihack",
        task="MiniHack-Room-5x5-v0",
        config=config,
        render_mode=None
    )

    obs, info = env.reset()

    # Find the NLELanguageWrapper
    # Start with gym_env since GymV21CompatibilityV0 wraps the actual environment
    current_env = env.gym_env if hasattr(env, 'gym_env') else env
    nle_wrapper = None
    while hasattr(current_env, 'env'):
        if current_env.__class__.__name__ == 'NLELanguageWrapper':
            nle_wrapper = current_env
            break
        current_env = current_env.env

    assert nle_wrapper is not None, "NLELanguageWrapper not found"
    assert nle_wrapper.include_lang_obs == include_lang_obs
    assert nle_wrapper.vlm == vlm_mode

    # Check observation content
    text_obs = obs.get("text", {})
    long_term_context = text_obs.get("long_term_context", "")
    has_lang_obs = "language observation" in long_term_context

    # Language observation presence should match the toggle regardless of VLM mode
    assert has_lang_obs == include_lang_obs, \
        f"Language observation presence should be {include_lang_obs} regardless of VLM mode"

    # In VLM mode, image should be present; otherwise None
    if vlm_mode:
        assert obs.get("image") is not None, "Image should be present in VLM mode"

    env.close()
