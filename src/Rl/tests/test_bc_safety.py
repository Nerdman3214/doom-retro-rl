import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_module("train_vizdoom_video_behavior_clone", "training/train_vizdoom_video_behavior_clone.py")
watcher = load_module("watch_vizdoom_z_breadcrumb_agent", "tools/watch_vizdoom_z_breadcrumb_agent.py")


def test_safe_route_action_filter_rejects_forbidden_controls():
    simple = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    use_action = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    assert trainer.is_safe_route_action(simple)
    assert not trainer.is_safe_route_action(use_action)


def test_bc_override_cooldown_blocks_immediate_repeats():
    assert watcher.bc_override_is_allowed(None, None, cooldown_steps=4)
    assert watcher.bc_override_is_allowed(0, 0, cooldown_steps=4) is False
    assert watcher.bc_override_is_allowed(3, 0, cooldown_steps=4) is False
    assert watcher.bc_override_is_allowed(4, 0, cooldown_steps=4)
