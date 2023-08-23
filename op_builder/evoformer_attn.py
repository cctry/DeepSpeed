# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder, installed_cuda_version
from deepspeed.accelerator import get_accelerator
import os
import yaml


class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_RANDOM_LTD"
    NAME = "evoformer_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
        self.cutlass_path = os.environ.get('CUTLASS_PATH')

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def sources(self):
        src_dir = 'csrc/deepspeed4science/evoformer_attn'
        return [f'{src_dir}/attention.cpp', f'{src_dir}/attention_back.cu', f'{src_dir}/attention.cu']

    def is_compatible(self, verbose=True):
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile kernels")
            return False
        if self.cutlass_path is None:
            self.warning("Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH")
            return False
        with open(f'{self.cutlass_path}/CITATION.cff', 'r') as f:
            version = yaml.safe_load(f)['version']
            if int(version.split('.')[0]) < 3:
                self.warning("Please use CUTLASS version >= 3.0.0")
                return False
        cuda_okay = True
        if not self.is_rocm_pytorch() and get_accelerator().is_available():
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = get_accelerator().get_device_properties(0).major
            if cuda_capability < 7:
                self.warning("Please use a GPU with compute capability >= 7.0")
                cuda_okay = False
            if torch_cuda_major < 11 or sys_cuda_major < 11:
                self.warning("Please use CUDA 11+")
                cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def include_paths(self):
        includes = [f'{self.cutlass_path}/include', f'{self.cutlass_path}/tools/util/include']
        return includes
