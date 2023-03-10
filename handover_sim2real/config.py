# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import easysim

from handover.config import cfg
from yacs.config import CfgNode as CN

_C = cfg

_C_handover_config = _C.clone()

# ---------------------------------------------------------------------------- #
# Policy config
# ---------------------------------------------------------------------------- #
_C.POLICY = CN()


_C.POLICY.TIME_ACTION_REPEAT = 0.15

_C.POLICY.TIME_CLOSE_GRIPPER = 0.5

_C.POLICY.BACK_STEP_SIZE = 0.03

_C.POLICY.POINT_STATE_YCB_RATIO = 0.875


def get_cfg(handover_config_only=False):
    if not handover_config_only:
        cfg = _C
    else:
        cfg = _C_handover_config
    return cfg.clone()
