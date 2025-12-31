import logging
from enum import Enum
from typing import Dict

import torch
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp


class Group(Enum):
    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


try:
    from libth_transformer.rtp_llm_ops import RtpProcessGroup, RtpProcessGroupType

    _group_map: Dict[Group, RtpProcessGroup] = {}

    def _group_type_convert(group: Group) -> RtpProcessGroupType:
        if group == Group.DP:
            return RtpProcessGroupType.DP_GROUP
        elif group == Group.TP:
            return RtpProcessGroupType.TP_GROUP
        elif group == Group.DP_AND_TP:
            return RtpProcessGroupType.DP_AND_TP_GROUP
        raise ValueError(f"Invalid group: {group}")

    def _get_group(group: Group) -> RtpProcessGroup:
        if group not in _group_map:
            _group_map[group] = RtpProcessGroup(_group_type_convert(group))
        return _group_map[group]

    def send(tensor: torch.Tensor, dst: int, group: Group = Group.DP_AND_TP) -> None:
        _get_group(group).send([tensor], dst)

    def recv(
        tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        return _get_group(group).recv([tensor], src)

    def broadcast(
        tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP
    ) -> None:
        _get_group(group).broadcast([tensor], src)

    def all_reduce(
        tensor: torch.Tensor, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        return _get_group(group).all_reduce([tensor])[0]

    def all_gather(
        tensor: torch.Tensor, group: Group = Group.DP_AND_TP
    ) -> torch.Tensor:
        return _get_group(group).all_gather([tensor])[0]

except ImportError:
    import os
    import sys
    def print_env_info():
        print("-" * 30)
        print("--- 运行环境检查 ---")
        # 打印 PYTHONPATH
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
        # 打印 sys.path (Python 实际搜索路径)
        print(f"sys.path: {sys.path}")
        # 打印 LD_LIBRARY_PATH
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
        print("-" * 30)

    print_env_info()
    logging.info("RtpProcessGroup not available, skipped. Defining dummy functions.")

    def _raise_error_on_call(*args, **kwargs):
        raise ImportError(
            "RtpProcessGroup functions are not available because 'libth_transformer' could not be imported."
        )

    send = recv = broadcast = all_reduce = all_gather = _raise_error_on_call
