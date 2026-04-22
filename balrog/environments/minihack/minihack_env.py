from typing import Optional

import gym
import minihack  # NOQA: F401
from omegaconf import OmegaConf

from balrog.environments.nle import AutoMore, NLELanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

MINIHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "MiniHack":
        MINIHACK_ENVS.append(id)


def _select_hide_obs_when_image(config) -> bool:
    try:
        return bool(OmegaConf.select(config, "eval.evolve.hide_obs_when_image", default=False))
    except Exception:
        try:
            return bool(config.eval.evolve.hide_obs_when_image)
        except Exception:
            return False


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
    minihack_kwargs = dict(config.envs.minihack_kwargs)
    skip_more = minihack_kwargs.pop("skip_more", False)
    vlm = True if config.agent.max_image_history > 0 else False
    include_lang_obs = minihack_kwargs.pop("include_lang_obs", True)
    include_perc_obs = minihack_kwargs.pop("include_perc_obs", False)
    use_textual_desc = minihack_kwargs.pop("use_textual_desc", False)
    hide_obs_when_image = _select_hide_obs_when_image(config)
    env = gym.make(
        task,
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
            "screen_descriptions",
        ],
        **minihack_kwargs,
    )
    if skip_more:
        env = AutoMore(env)
    env = NLELanguageWrapper(env, vlm=vlm, include_lang_obs=include_lang_obs, include_perc_obs=include_perc_obs, use_textual_desc=use_textual_desc, hide_obs_when_image=hide_obs_when_image)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
