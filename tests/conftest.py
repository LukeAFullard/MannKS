import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_numpy_rng():
    """Make tests that draw from numpy's global RNG deterministic."""
    np.random.seed(20240101)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (decomposition auto mode, Monte Carlo checks)")
