# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import socket
from datetime import timedelta

import torch


def stateless_init_process_group(host: str, port: int, rank: int, world_size: int, device: int):
    from torch.distributed import TCPStore

    try:
        from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
        from sglang.srt.distributed.utils import StatelessProcessGroup
    except ImportError as exc:
        raise ModuleNotFoundError("Asynchronous OPD NCCL transport requires SGLang's PyNcclCommunicator") from exc

    is_master = rank == 0
    if is_master:
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((host, port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    else:
        listen_socket = None
        listen_fd = None
    store = TCPStore(
        host_name=host,
        port=port,
        world_size=world_size,
        is_master=is_master,
        timeout=timedelta(seconds=300),
        use_libuv=False,
        master_listen_fd=listen_fd,
    )
    try:
        process_group = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            socket=listen_socket,
            data_expiration_seconds=3600,
        )
    except TypeError:
        process_group = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=3600,
        )
        process_group._listen_socket = listen_socket
    communicator = PyNcclCommunicator(process_group, device=device)
    if getattr(communicator, "available", True) and hasattr(communicator, "disabled"):
        communicator.disabled = False
    return communicator


def broadcast_hidden_chunk(tensor: torch.Tensor, communicator) -> None:
    communicator.broadcast(tensor, src=0, stream=torch.cuda.current_stream())
    torch.cuda.synchronize(tensor.device)
