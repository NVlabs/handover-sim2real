# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import os
import sys


def add_sys_path_from_env(name):
    assert name in os.environ, "Environment variable '{}' is not set".format(name)
    if os.environ[name] not in sys.path:
        sys.path.append(os.environ[name])
