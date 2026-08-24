"""Trivial suite: proves the package installs and imports cleanly.

This is the floor other tickets build test coverage on top of -- it is not
meant to exercise architecture, since none is built yet (ticket #79).
"""

import torch
import mujoco

import patchworks


def test_package_imports():
    assert patchworks.__version__ == "0.0.0"


def test_torch_is_pinned_and_cpu_only():
    assert torch.__version__.startswith("2.2.2")
    # The declared compute target is CPU; this asserts no CUDA-only path is
    # required to import or use torch on the development laptop.
    assert torch.cuda.is_available() is False


def test_mujoco_is_pinned():
    assert mujoco.__version__ == "3.10.0"
