#!/usr/bin/env python3
"""Direct-main tests for observational-only CEM query-tape recording."""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from spider.config import Config
from spider.optimizers.sampling import make_rollout_fn
from spider.query_tape import (
    cem_query_tape_chunk_count,
    finalize_cem_query_tape,
    record_cem_query_chunk,
)


def test_config_defaults_off() -> None:
    """Query recording must be inert unless explicitly enabled."""
    config = Config()
    assert config.query_tape_enabled is False
    assert config.query_tape_output_dir == ""
    assert config.query_tape_run_id == ""
    assert config.query_tape_max_chunks == 0
    assert config.query_tape_stop_after_chunks == 0
    assert config.query_tape_record_start_sim_step == 0
    assert config.query_tape_record_geometry_state is False


def test_record_chunk_schema_and_counter() -> None:
    """Recorder must persist deterministic arrays and monotonic chunk IDs."""
    with tempfile.TemporaryDirectory(prefix="e182_query_tape_test_") as directory:
        config = SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=directory,
            query_tape_run_id="unit",
        )
        payload = {
            "qpos": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            "rewards": torch.tensor([1.0, 2.0]),
            "selected_indices": torch.tensor([1], dtype=torch.int64),
            "sample_gate_valid_mask": torch.tensor([False, True]),
            "sample_body_gate_valid_mask": torch.tensor([True, True]),
            "sample_hand_gate_valid_mask": torch.tensor([False, True]),
            "sample_leg_gate_valid_mask": torch.tensor([True, False]),
        }
        first = record_cem_query_chunk(config, payload)
        second = record_cem_query_chunk(config, payload)

        assert first["chunk_index"] == 0
        assert second["chunk_index"] == 1
        first_path = Path(first["path"])
        assert first_path.name == "chunk_000000.npz"
        arrays = np.load(first_path)
        assert arrays["qpos"].shape == (2, 3, 4)
        assert arrays["selected_indices"].tolist() == [1]
        assert arrays["sample_gate_valid_mask"].tolist() == [False, True]
        manifest = json.loads(
            (Path(directory) / "unit" / "chunk_manifest.json").read_text()
        )
        assert manifest["status"] == "RECORDING"
        assert manifest["chunk_count"] == 2
        assert len(manifest["chunks"][0]["sha256"]) == 64

        complete = finalize_cem_query_tape(
            config,
            provenance={"case_id": "unit_case", "seed": 0},
        )
        assert complete["status"] == "COMPLETE"
        assert complete["chunk_count"] == 2
        assert len(complete["content_sha256"]) == 64
        assert complete["provenance"]["case_id"] == "unit_case"
        assert finalize_cem_query_tape(config) == complete


def test_deterministic_mock_rollout_exact() -> None:
    """Instrumentation must be an exact no-op on a deterministic backend."""

    class MockEnv:
        def __init__(self) -> None:
            self.state = torch.zeros((2, 3), dtype=torch.float32)
            self.saved = self.state.clone()

    def step_env(config, env, ctrl):
        del config
        env.state = env.state + ctrl

    def save_state(env):
        env.saved = env.state.clone()
        return None

    def load_state(env, state):
        del state
        env.state = env.saved.clone()
        return env

    def get_reward(config, env, ref):
        del config, ref
        reward = env.state.sum(dim=1)
        return reward, {
            "mock_metric": reward.clone(),
            "robot_object_penalty": torch.zeros_like(reward),
            "leg_object_penalty": torch.zeros_like(reward),
            "surface_band_rew": torch.zeros_like(reward),
            "surface_band_gate": torch.ones_like(reward),
            "surface_band_decay_factor": torch.ones_like(reward),
        }

    def get_terminate(config, env, ref):
        del config, ref
        return torch.zeros(env.state.shape[0], dtype=torch.bool)

    def get_trace(config, env):
        del config
        return torch.zeros((env.state.shape[0], 0, 3), dtype=torch.float32)

    def save_env_params(config, env):
        del config, env
        return None

    def load_env_params(config, env, params):
        del config, params
        return env

    def copy_sample_state(config, env, good, bad):
        del config, env, good, bad

    def get_qpos(config, env):
        del config
        return env.state

    def get_geometry_state(config, env):
        del config
        return {"mock_xpos": env.state[:, None, :]}

    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
        get_qpos,
        get_geometry_state,
    )
    config_off = Config()
    config_off.device = "cpu"
    config_off.nq = 3
    config_off.num_samples = 2
    config_off.query_tape_enabled = False
    config_on = deepcopy(config_off)
    config_on.query_tape_enabled = True
    config_on.query_tape_record_geometry_state = True
    ctrls = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 100.0
    ref_slice = (torch.zeros(4),)
    outputs = []
    for config in (config_off, config_on):
        outputs.append(rollout(config, MockEnv(), ctrls.clone(), ref_slice, {}))
    off_ctrls, off_reward, off_terminate, off_info = outputs[0]
    on_ctrls, on_reward, on_terminate, on_info = outputs[1]
    assert torch.equal(off_ctrls, on_ctrls)
    assert torch.equal(off_reward, on_reward)
    assert torch.equal(off_terminate, on_terminate)
    assert "_query_tape_qpos" not in off_info
    assert on_info["_query_tape_qpos"].shape == (2, 4, 3)
    assert on_info["_query_tape_geometry_qpos"].shape == (2, 4, 3)
    assert on_info["_query_tape_geometry_mock_xpos"].shape == (2, 4, 1, 3)
    expected_post_step = torch.cumsum(ctrls, dim=1)
    expected_pre_step = torch.cat(
        [torch.zeros_like(expected_post_step[:, :1]), expected_post_step[:, :-1]],
        dim=1,
    )
    assert torch.equal(on_info["_query_tape_qpos"], expected_post_step)
    assert torch.equal(on_info["_query_tape_geometry_qpos"], expected_pre_step)
    trace_keys = {
        "_query_tape_qpos",
        "_query_tape_geometry_qpos",
        "_query_tape_geometry_mock_xpos",
        "_query_tape_trace_robot_object_penalty",
        "_query_tape_trace_leg_object_penalty",
        "_query_tape_trace_surface_band_rew",
        "_query_tape_trace_surface_band_gate",
        "_query_tape_trace_surface_band_decay_factor",
    }
    assert set(off_info) == set(on_info) - trace_keys
    for key in trace_keys - {
        "_query_tape_qpos",
        "_query_tape_geometry_qpos",
        "_query_tape_geometry_mock_xpos",
    }:
        assert on_info[key].shape == (2, 4)
    for key in off_info:
        assert torch.equal(off_info[key], on_info[key])


def test_record_chunk_cap_is_observational_only() -> None:
    """A positive cap persists only the requested number of chunks."""
    with tempfile.TemporaryDirectory(prefix="e182_query_tape_cap_") as directory:
        config = SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=directory,
            query_tape_run_id="cap",
            query_tape_max_chunks=1,
        )
        payload = {
            "qpos": np.zeros((2, 3, 4), dtype=np.float32),
            "rewards": np.zeros(2, dtype=np.float32),
        }
        first = record_cem_query_chunk(config, payload)
        skipped = record_cem_query_chunk(config, payload)
        assert first["chunk_index"] == 0
        assert skipped == {"status": "SKIPPED_MAX_CHUNKS", "chunk_count": 1}
        assert cem_query_tape_chunk_count(config) == 1
        complete = finalize_cem_query_tape(config)
        assert complete["status"] == "COMPLETE"
        assert complete["chunk_count"] == 1


def test_record_start_skips_early_artifacts() -> None:
    """A record-start gate does not create a tape before the frozen sim step."""
    with tempfile.TemporaryDirectory(prefix="e182_query_tape_start_") as directory:
        config = SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=directory,
            query_tape_run_id="start",
            query_tape_max_chunks=1,
            query_tape_record_start_sim_step=12,
            _query_tape_current_sim_step=10,
        )
        payload = {
            "qpos": np.zeros((2, 3, 4), dtype=np.float32),
            "rewards": np.zeros(2, dtype=np.float32),
        }
        skipped = record_cem_query_chunk(config, payload)
        assert skipped["status"] == "SKIPPED_BEFORE_START"
        assert not (Path(directory) / "start").exists()
        config._query_tape_current_sim_step = 12
        recorded = record_cem_query_chunk(config, payload)
        assert recorded["chunk_index"] == 0


def main() -> int:
    """Run recorder tests without pytest discovery."""
    tests = (
        test_config_defaults_off,
        test_record_chunk_schema_and_counter,
        test_deterministic_mock_rollout_exact,
        test_record_chunk_cap_is_observational_only,
        test_record_start_skips_early_artifacts,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_QUERY_TAPE_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
