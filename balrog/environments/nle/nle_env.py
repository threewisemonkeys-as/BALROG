from typing import Optional

import gym
import nle  # NOQA: F401
from omegaconf import OmegaConf

from balrog.environments.nle import AutoMore, NLELanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

NETHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if "NetHack" in id:
        NETHACK_ENVS.append(id)


def _select_hide_obs_when_image(config) -> bool:
    try:
        return bool(OmegaConf.select(config, "eval.evolve.hide_obs_when_image", default=False))
    except Exception:
        try:
            return bool(config.eval.evolve.hide_obs_when_image)
        except Exception:
            return False


def make_nle_env(env_name, task, config, render_mode: Optional[str] = None):
    nle_kwargs = dict(config.envs.nle_kwargs)
    skip_more = nle_kwargs.pop("skip_more", False)
    include_lang_obs = nle_kwargs.pop("include_lang_obs", True)
    include_perc_obs = nle_kwargs.pop("include_perc_obs", False)
    use_textual_desc = nle_kwargs.pop("use_textual_desc", False)
    vlm = True if config.agent.max_image_history > 0 else False
    hide_obs_when_image = _select_hide_obs_when_image(config)
    env = gym.make(task, **nle_kwargs)
    if skip_more:
        env = AutoMore(env)
    env = NLELanguageWrapper(env, vlm=vlm, include_lang_obs=include_lang_obs, include_perc_obs=include_perc_obs, use_textual_desc=use_textual_desc, hide_obs_when_image=hide_obs_when_image)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
