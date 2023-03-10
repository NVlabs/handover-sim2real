# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

"""Handover-Sim2Real package."""

DISTRIBUTION_NAME = "handover-sim2real"


# NOTE (roflaherty): This is inspired by how matplotlib generates its version value.
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L161
def _get_version() -> str:
    """Return the version string used for __version__."""
    # Standard Library
    import pathlib

    # Get path to project root (i.e. the directory where the .git folder lives)
    root = pathlib.Path(__file__).resolve().parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        this_version = _get_version_from_git_tag(root.as_posix())
    else:  # Get the version from the _version.py setuptools_scm file.
        this_version = _get_version_from_setuptools_scm_file()

    return this_version


def _get_version_from_git_tag(root: str) -> str:
    """Return the version string based on the git commit and the latest git tag."""
    # Third Party
    import setuptools_scm

    this_version: str
    # See the `setuptools_scm` documentation for the description of the schemes used below.
    # https://pypi.org/project/setuptools-scm/
    # NOTE: If these values are updated, they need to be also updated in `pyproject.toml`.
    this_version = setuptools_scm.get_version(
        root=root,
        version_scheme="no-guess-dev",
        local_scheme="dirty-tag",
    )
    return this_version


def _get_version_from_setuptools_scm_file() -> str:
    """Return the version string based on the latest installed package version."""
    try:
        # Standard Library
        from importlib.metadata import version
    except ModuleNotFoundError:
        # NOTE: `importlib.metadata` is provisional in Python 3.9 and standard in Python 3.10.
        # `importlib_metadata` is the back ported library for older versions of python.
        # Third Party
        from importlib_metadata import version  # type: ignore[no-redef]

    this_version = version(DISTRIBUTION_NAME)
    return this_version


# Set `__version__` attribute
__version__ = _get_version()

# Remove `_get_version` so it is not added as an attribute
del _get_version

from gym.envs.registration import register

register(
    id="HandoverSim2RealTrainEnv-v1",
    entry_point="handover_sim2real.train_env:HandoverSim2RealTrainEnv",
)
