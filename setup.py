from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='cdist',
    ext_modules=[
        CUDAExtension('sparse_cdist', [
            'sparse_cdist_cuda.cpp',
            'sparse_cdist_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
