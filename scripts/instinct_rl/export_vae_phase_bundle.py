#!/usr/bin/env python3
# Copyright (c) 2025 WBCHSI / InstinctLab. SPDX-License-Identifier: BSD-3-Clause
"""Offline export: split a full Instinct-RL ``model_*.pt`` into ``vae_phase_bundle_*.pt`` (v1 schema).

Does not import Isaac Sim. Only reads PyTorch checkpoints and optional ``params/agent.yaml`` next to a run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def strip_prefix_state_dict(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Keep keys starting with ``prefix`` and remove that prefix from the result keys."""
    plen = len(prefix)
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


def unwrap_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """If every key starts with ``module.``, strip that layer (DDP / legacy wrapping)."""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[8:]: state_dict[k] for k in keys}
    return state_dict


def try_load_agent_yaml(agent_yaml: Path | None, run_dir: Path) -> dict[str, Any]:
    """Load ``policy`` subsection fields useful for meta (vae_* component lists)."""
    path = agent_yaml
    if path is None:
        candidate = run_dir / "params" / "agent.yaml"
        if candidate.is_file():
            path = candidate
    if path is None or not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        root = yaml.safe_load(text)
    except yaml.constructor.ConstructorError:
        # Isaac Lab ``dump_yaml`` can emit ``!!python/tuple`` and similar tags;
        # ``safe_load`` rejects those. Only use for your own training run output.
        root = yaml.unsafe_load(text)
    except yaml.YAMLError as e:
        print(f"[WARN]: Could not parse agent yaml {path}: {e}. Meta vae_* fields will be missing.")
        return {}
    if not isinstance(root, dict):
        return {}
    policy = root.get("policy")
    return policy if isinstance(policy, dict) else {}


def build_bundle(
    ckpt: dict[str, Any],
    *,
    source_checkpoint: str,
    agent_policy: dict[str, Any],
    latent_size_override: int | None,
) -> dict[str, Any]:
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing key 'model_state_dict'.")
    ms = ckpt["model_state_dict"]
    if not isinstance(ms, dict):
        raise TypeError("model_state_dict must be a dict.")
    ms = unwrap_module_prefix(ms)

    bundle: dict[str, Any] = {"format_version": 1}

    enc = strip_prefix_state_dict(ms, "encoders.")
    if enc:
        bundle["encoders_state_dict"] = enc

    dec = strip_prefix_state_dict(ms, "actor.decoder.")
    if not dec:
        raise ValueError("No keys with prefix 'actor.decoder.' in model_state_dict (not a VAE-style actor?).")

    bundle["mlp_vae_decoder_state_dict"] = dec
    bundle["mlp_vae_encoder_state_dict"] = strip_prefix_state_dict(ms, "actor.encoder.")

    prior = strip_prefix_state_dict(ms, "actor.prior_net.")
    bundle["mlp_vae_prior_net_state_dict"] = prior if prior else None

    if "policy_normalizer_state_dict" in ckpt:
        bundle["policy_normalizer_state_dict"] = ckpt["policy_normalizer_state_dict"]

    iter_val = ckpt.get("iter")
    try:
        iter_int = int(iter_val) if iter_val is not None else None
    except (TypeError, ValueError):
        iter_int = None

    latent_size = latent_size_override
    if latent_size is None and isinstance(agent_policy.get("vae_latent_size"), (int, float)):
        latent_size = int(agent_policy["vae_latent_size"])
    if latent_size is None:
        # Encoder last layer output is 2 * latent for mean/log_std split (MlpVae)
        enc_sd = bundle.get("mlp_vae_encoder_state_dict") or {}
        for k in sorted(enc_sd.keys()):
            if k.endswith("weight") and ".net." in k and enc_sd[k].ndim == 2:
                out_dim = enc_sd[k].shape[0]
                if out_dim > 0 and out_dim % 2 == 0:
                    latent_size = out_dim // 2
                    break

    has_prior = bundle["mlp_vae_prior_net_state_dict"] is not None

    meta: dict[str, Any] = {
        "iter": iter_int,
        "source_checkpoint": os.path.abspath(source_checkpoint),
        "actor_critic_inferred_from_keys": [
            "has_encoders_prefix=" + str(bool(enc)),
            "has_actor_decoder=" + str(bool(dec)),
            "has_prior_net=" + str(has_prior),
        ],
    }
    if latent_size is not None:
        meta["latent_size"] = latent_size
    meta["has_prior_net"] = has_prior

    for k in ("vae_input_subobs_components", "vae_aux_subobs_components", "vae_prior_subobs_components"):
        if k in agent_policy:
            v = agent_policy[k]
            meta[k] = list(v) if v is not None else None

    bundle["meta"] = meta
    return bundle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt from Instinct-RL training.")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Default: same dir as checkpoint, vae_phase_bundle_<iter>.pt",
    )
    p.add_argument(
        "--agent-yaml",
        type=str,
        default=None,
        help="Optional path to params/agent.yaml for vae_* meta fields. "
        "Default: <checkpoint_dir>/params/agent.yaml if it exists.",
    )
    p.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help="Optional override for meta latent_size (otherwise from yaml or encoder weight shape).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if encoders or encoder/prior shards are empty when you expect a full EncoderVae checkpoint.",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        print("ERROR: checkpoint must be a dict.", file=sys.stderr)
        return 1

    run_dir = ckpt_path.parent
    agent_yaml = Path(args.agent_yaml).expanduser().resolve() if args.agent_yaml else None
    agent_policy = try_load_agent_yaml(agent_yaml, run_dir)

    iter_val = ckpt.get("iter")
    try:
        iter_tag = int(iter_val) if iter_val is not None else None
    except (TypeError, ValueError):
        iter_tag = None
    iter_str = str(iter_tag) if iter_tag is not None else "unknown"

    out_path = args.output
    if out_path is None:
        out_path = str(run_dir / f"vae_phase_bundle_{iter_str}.pt")
    else:
        out_path = str(Path(out_path).expanduser().resolve())

    try:
        bundle = build_bundle(
            ckpt,
            source_checkpoint=str(ckpt_path),
            agent_policy=agent_policy,
            latent_size_override=args.latent_size,
        )
    except (KeyError, ValueError, TypeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.strict:
        ms = unwrap_module_prefix(ckpt["model_state_dict"])
        if not any(k.startswith("encoders.") for k in ms):
            print("ERROR [--strict]: no model_state_dict keys with prefix 'encoders.'.", file=sys.stderr)
            return 1
        if not any(k.startswith("actor.encoder.") for k in ms):
            print("ERROR [--strict]: no model_state_dict keys with prefix 'actor.encoder.'.", file=sys.stderr)
            return 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bundle, out_path)
    print(f"[INFO] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
