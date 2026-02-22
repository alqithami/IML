from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SSDEnvError(RuntimeError):
    pass


@dataclass
class EnvSpec:
    name: str
    module: str
    cls: str


SSD_SPECS = {
    "cleanup": EnvSpec(name="cleanup", module="social_dilemmas.envs.cleanup", cls="CleanupEnv"),
    "harvest": EnvSpec(name="harvest", module="social_dilemmas.envs.harvest", cls="HarvestEnv"),
}


def _try_instantiate(env_cls, *, num_agents: int, seed: Optional[int], env_kwargs: Dict[str, Any]):
    """Best-effort constructor calling across possible signatures."""
    # Common constructor kwargs seen across multi-agent envs
    candidates = [
        dict(num_agents=num_agents, **env_kwargs),
        dict(n_agents=num_agents, **env_kwargs),
        dict(num_agents=num_agents),
        dict(n_agents=num_agents),
        dict(**env_kwargs),
        dict(),
    ]

    last_exc = None
    for kwargs in candidates:
        try:
            env = env_cls(**kwargs)  # type: ignore[arg-type]
            # Seed if supported
            if seed is not None:
                if hasattr(env, "seed") and callable(getattr(env, "seed")):
                    try:
                        env.seed(seed)
                    except Exception:
                        pass
                np.random.seed(seed)
            return env
        except Exception as e:
            last_exc = e
            continue

    raise SSDEnvError(
        f"Failed to instantiate {env_cls} with tried kwargs variants. Last error: {last_exc}"
    )


def make_ssd_env(env_name: str, num_agents: int, seed: Optional[int] = None, **env_kwargs):
    env_name = env_name.lower().strip()
    if env_name not in SSD_SPECS:
        raise SSDEnvError(f"Unknown SSD env '{env_name}'. Choose from {list(SSD_SPECS.keys())}")
    spec = SSD_SPECS[env_name]

    try:
        mod = importlib.import_module(spec.module)
    except Exception as e:
        raise SSDEnvError(
            f"Could not import '{spec.module}'. Make sure SSD is installed (social_dilemmas). Error: {e}"
        )
    if not hasattr(mod, spec.cls):
        raise SSDEnvError(f"Module '{spec.module}' has no attribute '{spec.cls}'.")

    env_cls = getattr(mod, spec.cls)
    env = _try_instantiate(env_cls, num_agents=num_agents, seed=seed, env_kwargs=env_kwargs)
    return env


def get_agent_ids(obs_dict: Dict[Any, Any]) -> List[str]:
    """Return agent ids as strings, stable order."""
    # obs_dict keys can be strings already, or ints.
    keys = list(obs_dict.keys())
    # Filter out RLlib's special __all__ if present
    keys = [k for k in keys if k != "__all__"]
    return [str(k) for k in keys]


def get_action_space_n(env) -> int:
    """Best effort to find a discrete action space size."""
    # SSD envs often expose env.action_space as gym.spaces.Discrete or dict-like
    asp = getattr(env, "action_space", None)
    if asp is None:
        raise SSDEnvError("Environment has no action_space.")
    if hasattr(asp, "n"):
        return int(asp.n)
    # Dict of spaces
    if isinstance(asp, dict):
        first = next(iter(asp.values()))
        if hasattr(first, "n"):
            return int(first.n)
    # PettingZoo style
    if hasattr(env, "action_spaces"):
        spaces = getattr(env, "action_spaces")
        if isinstance(spaces, dict):
            first = next(iter(spaces.values()))
            if hasattr(first, "n"):
                return int(first.n)
    raise SSDEnvError(f"Could not infer discrete action space size from action_space={type(asp)}")


def _extract_rgb_obs(obs_entry: Any) -> np.ndarray:
    """Extract an RGB array (H,W,C) from SSD per-agent observations.

    SSD forks/versions may return per-agent obs as:
      - np.ndarray (H,W,C)
      - dict containing an array under keys like 'curr_obs', 'rgb', 'image', etc.
      - nested dicts
      - tuples/lists containing arrays

    This helper makes the pipeline robust to all of the above.
    """
    if isinstance(obs_entry, np.ndarray):
        return obs_entry

    if isinstance(obs_entry, (list, tuple)):
        # Try each entry until we find an ndarray
        for v in obs_entry:
            try:
                arr = _extract_rgb_obs(v)
                if isinstance(arr, np.ndarray):
                    return arr
            except Exception:
                pass

    if isinstance(obs_entry, dict):
        # Common keys used in dict observation spaces
        for k in ("curr_obs", "rgb", "image", "observation", "obs", "state", "board", "pixels"):
            v = obs_entry.get(k, None)
            if isinstance(v, np.ndarray):
                return v

        # Fallback: first 3D ndarray value
        for v in obs_entry.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v

        # Fallback: recurse into nested dicts
        for v in obs_entry.values():
            if isinstance(v, dict):
                try:
                    return _extract_rgb_obs(v)
                except Exception:
                    pass

    raise SSDEnvError(
        f"Unsupported observation type {type(obs_entry)}. "
        f"Expected np.ndarray or dict/list/tuple containing an ndarray."
    )


def preprocess_obs(obs: Any) -> np.ndarray:
    """Convert SSD obs (array or dict containing array) to float32 in [0,1]."""
    obs_arr = _extract_rgb_obs(obs)

    if obs_arr.dtype != np.float32:
        obs_arr = obs_arr.astype(np.float32)

    # Many SSD observations are uint8 0..255
    try:
        if obs_arr.max() > 1.5:
            obs_arr = obs_arr / 255.0
    except Exception:
        pass

    return obs_arr
