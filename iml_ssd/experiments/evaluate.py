from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from iml_ssd.config import load_yaml
from iml_ssd.envs.ssd_env import get_action_space_n, get_agent_ids, make_ssd_env, preprocess_obs
from iml_ssd.institution.iml_wrapper import IMLConfig, IMLWrapper
from iml_ssd.institution.rules import HighWasteNoCleanRule, LowAppleDensityHarvestRule, NoPunishmentBeamRule
from iml_ssd.rl.networks import SharedCNNActorCritic
from iml_ssd.utils.logging import CSVLogger
from iml_ssd.utils.metrics import gini


def _build_iml_config(cfg: Dict[str, Any]) -> IMLConfig:
    iml_cfg = cfg.get("iml", {}) or {}
    enabled = bool(iml_cfg.get("enabled", False))

    rules: List[Any] = []
    rule_names = iml_cfg.get("rules", ["no_punishment_beam"])
    if isinstance(rule_names, str):
        rule_names = [rule_names]
    for r in rule_names:
        r = str(r)
        if r == "no_punishment_beam":
            rules.append(NoPunishmentBeamRule())
        elif r == "low_density_harvest":
            params = iml_cfg.get("low_density_harvest", {}) or {}
            rules.append(LowAppleDensityHarvestRule(**params))
        elif r == "high_waste_no_clean":
            params = iml_cfg.get("high_waste_no_clean", {}) or {}
            rules.append(HighWasteNoCleanRule(**params))
        else:
            raise ValueError(f"Unknown rule '{r}'.")

    return IMLConfig(
        enabled=enabled,
        p_detect_true=float(iml_cfg.get("p_detect_true", 0.9)),
        p_detect_false=float(iml_cfg.get("p_detect_false", 0.01)),
        sanction=float(iml_cfg.get("sanction", 0.5)),
        p_review=float(iml_cfg.get("p_review", 0.0)),
        write_ledger=False,
        ledger_path=None,
        rules=rules,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = load_yaml(cfg_path)

    seed = int(args.seed) if args.seed is not None else int(cfg.get("train", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    obs_shape = tuple(ckpt["obs_shape"])
    n_actions = int(ckpt["n_actions"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedCNNActorCritic(obs_shape=obs_shape, n_actions=n_actions).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env_name = str(cfg.get("env", {}).get("name", "cleanup")).lower()
    num_agents = int(cfg.get("env", {}).get("num_agents", 5))
    env_kwargs = cfg.get("env", {}).get("kwargs", {}) or {}
    env = make_ssd_env(env_name, num_agents=num_agents, seed=seed, **env_kwargs)

    # Fixed-horizon episodes (SSD forks may not signal termination reliably)
    max_episode_len = int(cfg.get('env', {}).get('max_episode_len', 2000))

    iml_cfg = _build_iml_config(cfg)
    if iml_cfg.enabled:
        env = IMLWrapper(env, iml_cfg, run_dir=None, seed=seed)

    out_csv = CSVLogger(run_dir / "eval.csv")

    for ep in range(int(args.episodes)):
        obs = env.reset()
        agent_ids = get_agent_ids(obs)
        ep_returns = {aid: 0.0 for aid in agent_ids}
        ep_len = 0
        iml_counts = dict(truth=0, detected=0, sanctions=0, false_pos=0, overturned=0)

        while True:
            obs_batch = np.stack([preprocess_obs(obs[aid]) for aid in agent_ids], axis=0)
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, values = model.forward(obs_t)
                actions = torch.argmax(logits, dim=-1)

            action_dict = {aid: int(actions[i].item()) for i, aid in enumerate(agent_ids)}
            next_obs, rewards, dones, infos = env.step(action_dict)

            ep_len += 1
            for aid in agent_ids:
                ep_returns[aid] += float(rewards.get(aid, 0.0))
                if isinstance(infos, dict) and aid in infos and isinstance(infos[aid], dict):
                    iml = infos[aid].get("iml")
                    if isinstance(iml, dict):
                        iml_counts["truth"] += int(bool(iml.get("truth_any", False)))
                        iml_counts["detected"] += int(bool(iml.get("detected_any", False)))
                        iml_counts["sanctions"] += int(iml.get("sanctions", 0) or 0)
                        iml_counts["false_pos"] += int(bool(iml.get("false_positive", False)))
                        iml_counts["overturned"] += int(bool(iml.get("overturned", False)))

            obs = next_obs

            episode_done = False
            if isinstance(dones, dict):
                episode_done = bool(dones.get("__all__", False))
                if not episode_done:
                    agent_keys = [k for k in dones.keys() if k != "__all__"]
                    if agent_keys:
                        episode_done = all(bool(dones[k]) for k in agent_keys)

            # Time-limit segmentation
            time_limit = False
            if (not episode_done) and ep_len >= max_episode_len:
                episode_done = True
                time_limit = True

            if episode_done:
                returns_list = [ep_returns[aid] for aid in agent_ids]
                out_csv.write({
                    "episode": ep,
                    "episode_len": ep_len,
                    "return_mean": float(np.mean(returns_list)),
                    "return_sum": float(np.sum(returns_list)),
                    "return_gini": gini(returns_list),
                    "iml_truth": iml_counts["truth"],
                    "iml_detected": iml_counts["detected"],
                    "iml_sanctions": iml_counts["sanctions"],
                    "iml_false_pos": iml_counts["false_pos"],
                    "iml_overturned": iml_counts["overturned"],
                    "time_limit": time_limit,
                })
                out_csv.flush()
                break

    out_csv.close()
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    print(f"Evaluation saved to {run_dir/'eval.csv'}")


if __name__ == "__main__":
    main()
