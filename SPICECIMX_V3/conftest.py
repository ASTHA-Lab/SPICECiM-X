"""
Ensure `spicecimx` is importable when running ``pytest`` from SPICECIMX_V3
even without ``pip install -e .``, and expose the repo locations tests need.

Layout::

    <REPO_TOP>/                 top-level project (config_cifar10.ini, supports/)
    <REPO_TOP>/SPICECIMX_V3/    this package root (PKG_ROOT)
"""
import os
import sys

import pytest

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_TOP = os.path.dirname(PKG_ROOT)
SUPPORTS_DIR = os.path.join(REPO_TOP, "supports")
SAMPLES_DIR = os.path.join(PKG_ROOT, "samples")

# Add the package root to sys.path so the package is discoverable
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


@pytest.fixture(scope="session")
def repo_top() -> str:
    """Top-level repo folder (holds supports/ and config_cifar10.ini)."""
    return REPO_TOP


@pytest.fixture(scope="session")
def supports_dir() -> str:
    """Top-level ``supports/`` folder (golden decks, CSV tables, peripherals)."""
    return SUPPORTS_DIR
