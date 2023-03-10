# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

"""Unit tests for the `handover-sim2real` package version."""

# Standard Library
import pathlib

# NVIDIA
import handover_sim2real


def test_handover_sim2real_version() -> None:
    """Test that the `handover-sim2real` package version is set."""
    assert handover_sim2real.__version__ is not None
    assert handover_sim2real.__version__ != ""


def test_handover_sim2real_get_version_from_git_tag() -> None:
    """Test that the `_get_version_from_git_tag` function."""
    root = pathlib.Path(__file__).resolve().parent.parent
    version = handover_sim2real._get_version_from_git_tag(root.as_posix())
    # The version can be anything, just want to check that it returns something and doesn't error
    # out.
    assert version is not None


def test_handover_sim2real_get_version_from_setuptools_scm_file() -> None:
    """Test that the `_get_version_from_setuptools_scm_file` function."""
    version = handover_sim2real._get_version_from_setuptools_scm_file()
    # The version can be anything, just want to check that it returns something and doesn't error
    # out.
    assert version is not None
