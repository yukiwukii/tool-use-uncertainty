"""Config loading, artefact paths and When2Tool imports shared by every step.

Every artefact is written as results/<step>/<name>.<cfg_hash>.<ext> so a changed
config never silently mixes with old outputs.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2T_ROOT = os.path.join(ROOT, "when2tool")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "qwen3_4b.yaml")

for p in (os.path.join(W2T_ROOT, "src"), W2T_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


class Config(dict):
    __getattr__ = dict.__getitem__

    @property
    def hash(self):
        blob = json.dumps(self, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:8]

    def path(self, step, name, ext):
        """results/<step>/<name>.<hash>.<ext>, creating the folder."""
        d = os.path.join(ROOT, self["results_dir"], step)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{name}.{self.hash}.{ext}")

    def data_path(self, split):
        return os.path.join(ROOT, self["data"][split])


def load_config(path=None):
    with open(path or os.environ.get("TOOLMECH_CONFIG", DEFAULT_CONFIG)) as f:
        cfg = Config(yaml.safe_load(f))
    seed_everything(cfg["seed"])
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


TIER = {"easy": 0, "medium": 1, "hard": 2}


def load_tasks(cfg, split):
    """Tasks for a split ('train', 'test', 'multihop_train', 'multihop_test')."""
    from utils import load_tasks_from_path  # When2Tool
    return load_tasks_from_path(cfg.data_path(split))


def task_env(task):
    envs = task.get("environments") or [{}]
    return envs[0].get("name", "unknown")
