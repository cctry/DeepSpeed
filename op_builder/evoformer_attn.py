from .builder import CUDAOpBuilder
import os

class EvoformerAttnBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_RANDOM_LTD"
    NAME = "evoformer_attn"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def sources(self):
        src_dir = 'csrc/deepspeed4science/evoformer_attn'
        return [
            f'{src_dir}/attention.cpp', 
            f'{src_dir}/attention_back.cu',
            f'{src_dir}/attention.cu'
        ]

    def include_paths(self):
        cutlass_path = os.environ.get('CUTLASS_PATH', "cutlass")
        includes = [f'{cutlass_path}/include', f'{cutlass_path}/tools/util/include']
        if self.is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            includes += ['{}/hiprand/include'.format(ROCM_HOME), '{}/rocrand/include'.format(ROCM_HOME)]
        return includes
