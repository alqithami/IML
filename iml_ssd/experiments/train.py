from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from iml_ssd.config import add_base_args, load_config_with_overrides, save_yaml
from iml_ssd.envs.ssd_env import get_action_space_n, get_agent_ids, make_ssd_env, preprocess_obs
from iml_ssd.institution.iml_wrapper import IMLConfig, IMLWrapper
from iml_ssd.institution.rules import HighWasteNoCleanRule, LowAppleDensityHarvestRule, NoPunishmentBeamRule
from iml_ssd.rl.networks import SharedCNNActorCritic
from iml_ssd.rl.ppo import PPOConfig, RolloutBuffer, compute_gae, ppo_update
from iml_ssd.utils.logging import RunLogger
from iml_ssd.utils.metrics import gini


def _build_iml_config(cfg: Dict[str, Any]) -> IMLConfig:
    iml_cfg = cfg.get("iml", {}) or {}
    enabled = bool(iml_cfg.get("enabled", False))

    # rules
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
        write_ledger=bool(iml_cfg.get("write_ledger", False)),
        ledger_path=iml_cfg.get("ledger_path", None),
        rules=rules,
    )


def _build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    ppo_cfg = cfg.get("ppo", {}) or {}
    return PPOConfig(
        total_steps=int(ppo_cfg.get("total_steps", 2_000_000)),
        rollout_steps=int(ppo_cfg.get("rollout_steps", 256)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        lr=float(ppo_cfg.get("lr", 2.5e-4)),
        num_epochs=int(ppo_cfg.get("num_epochs", 4)),
        minibatch_size=int(ppo_cfg.get("minibatch_size", 512)),
        clip_coef=float(ppo_cfg.get("clip_coef", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        target_kl=ppo_cfg.get("target_kl", None),
        device=str(ppo_cfg.get("device", "auto")),
    )


def main():
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    args = parser.parse_args()
    cfg, cfg_path = load_config_with_overrides(args)

    # Seeds
    seed = int(cfg.get("train", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_name = str(cfg.get("env", {}).get("name", "cleanup")).lower()
    num_agents = int(cfg.get("env", {}).get("num_agents", 5))

    # Run dir
    run_name = cfg.get("run", {}).get("name")
    if not run_name:
        condition = "iml" if bool(cfg.get("iml", {}).get("enabled", False)) else "baseline"
        run_name = f"{env_name}_{condition}_agents{num_agents}_seed{seed}"
    runs_dir = Path(cfg.get("run", {}).get("runs_dir", "runs"))
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist full resolved config for reproducibility
    save_yaml(cfg, run_dir / "config.yaml")

    logger = RunLogger(run_dir=run_dir)
    logger.save_config(cfg)

    # Env
    env_kwargs = cfg.get("env", {}).get("kwargs", {}) or {}
    env = make_ssd_env(env_name, num_agents=num_agents, seed=seed, **env_kwargs)

    iml_cfg = _build_iml_config(cfg)
    if iml_cfg.enabled:
        env = IMLWrapper(env, iml_cfg, run_dir=run_dir, seed=seed)

    obs = env.reset()
    agent_ids = get_agent_ids(obs)
    if len(agent_ids) == 0:
        raise RuntimeError("No agents returned by env.reset().")

    # Determine observation shape from first agent
    o0 = preprocess_obs(obs[agent_ids[0]])
    if o0.ndim != 3:
        raise RuntimeError(f"Expected RGB obs HxWxC, got shape {o0.shape}")
    obs_shape = tuple(map(int, o0.shape))  # H,W,C

    n_actions = get_action_space_n(env)

    ppo_cfg = _build_ppo_config(cfg)
    device = ppo_cfg.resolve_device()

    model = SharedCNNActorCritic(obs_shape=obs_shape, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    buf = RolloutBuffer(rollout_steps=ppo_cfg.rollout_steps, num_agents=len(agent_ids), obs_shape=obs_shape, device=device)

    # Episode tracking
    ep_returns = {aid: 0.0 for aid in agent_ids}
    ep_len = 0
    ep_idx = 0

    # IML counters
    iml_counts = dict(truth=0, detected=0, sanctions=0, false_pos=0, overturned=0)

    global_step = 0
    start_time = time.time()

    while global_step < ppo_cfg.total_steps:
        # Collect rollout
        buf.ptr = 0
        for t in range(ppo_cfg.rollout_steps):
            obs_batch = np.stack([preprocess_obs(obs[aid]) for aid in agent_ids], axis=0)  # (N,H,W,C)
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            with torch.no_grad():
                action_t, logp_t, entropy_t, value_t = model.get_action_and_value(obs_t)

            action_dict = {aid: int(action_t[i].item()) for i, aid in enumerate(agent_ids)}
            next_obs, rewards, dones, infos = env.step(action_dict)

            # Rewards/dones tensors
            r_t = torch.tensor([float(rewards.get(aid, 0.0)) for aid in agent_ids], dtype=torch.float32, device=device)

            # Determine done per agent
            if isinstance(dones, dict):
                if "__all__" in dones:
                    dflag = float(bool(dones["__all__"]))
                    d_t = torch.tensor([dflag] * len(agent_ids), dtype=torch.float32, device=device)
                    episode_done = bool(dones["__all__"])
                else:
                    d_t = torch.tensor([float(bool(dones.get(aid, False))) for aid in agent_ids], dtype=torch.float32, device=device)
                    episode_done = bool(d_t.max().item() > 0.0)
            else:
                d_t = torch.zeros((len(agent_ids),), dtype=torch.float32, device=device)
                episode_done = False

            # Store transition
            buf.add(
                obs=obs_t,
                actions=action_t,
                logprobs=logp_t,
                rewards=r_t,
                dones=d_t,
                values=value_t,
            )

            # Episode accounting + IML counters
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

            global_step += 1
            obs = next_obs

            if episode_done:
                # Log episode
                returns_list = [ep_returns[aid] for aid in agent_ids]
                ep_row = {
                    "env_step": global_step,
                    "episode_len": ep_len,
                    "return_mean": float(np.mean(returns_list)),
                    "return_sum": float(np.sum(returns_list)),
                    "return_gini": gini(returns_list),
                    "iml_truth": iml_counts["truth"],
                    "iml_detected": iml_counts["detected"],
                    "iml_sanctions": iml_counts["sanctions"],
                    "iml_false_pos": iml_counts["false_pos"],
                    "iml_overturned": iml_counts["overturned"],
                    "time_seconds": time.time() - start_time,
                }
                logger.log_episode(ep_idx, ep_row)
                ep_idx += 1
                ep_len = 0
                ep_returns = {aid: 0.0 for aid in agent_ids}
                iml_counts = dict(truth=0, detected=0, sanctions=0, false_pos=0, overturned=0)
                obs = env.reset()

            if global_step >= ppo_cfg.total_steps:
                break

        # Bootstrap last values for GAE
        with torch.no_grad():
            last_obs_batch = np.stack([preprocess_obs(obs[aid]) for aid in agent_ids], axis=0)
            last_obs_t = torch.tensor(last_obs_batch, dtype=torch.float32, device=device)
            last_values = model.get_value(last_obs_t)  # (N,)

        advantages, returns = compute_gae(
            rewards=buf.rewards,
            dones=buf.dones,
            values=buf.values,
            last_values=last_values,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        # PPO update
        train_metrics = ppo_update(model, optimizer, buf, advantages, returns, ppo_cfg)
        train_metrics["charts/steps"] = global_step
        train_metrics["charts/sps"] = float(global_step / max(1e-6, (time.time() - start_time)))

        logger.log_train(global_step, train_metrics)

    # Save model
    ckpt = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "obs_shape": obs_shape,
        "n_actions": n_actions,
        "agent_ids": agent_ids,
    }
    torch.save(ckpt, run_dir / "model.pt")
    logger.close()

    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    print(f"Done. Saved to: {run_dir}")


if __name__ == "__main__":
    main()
