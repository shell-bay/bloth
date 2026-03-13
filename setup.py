"""
Bloth: Ultra-Fast CUDA Kernels for LLM Training
================================================
A high-performance kernel library that surpasses Unsloth by leveraging:
- CUTLASS-based GEMM with warp specialization
- TMA (Tensor Memory Accelerator) for async data movement
- Software pipelining for optimal instruction overlap
- Automatic kernel fusion
- FP8/BF16/FP16 mixed precision training
- Support for any model architecture

Copyright 2026 Bloth Team. All rights reserved.
Licensed under Apache 2.0
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

# Get CUDA paths
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
CUDA_VERSION = os.environ.get('CUDA_VERSION', '')

# Compiler flags for maximum performance
NVCC_FLAGS = [
    '-O3',
    '--use_fast_math',
    '-gencode=arch=compute_80,code=sm_80',  # Ampere
    '-gencode=arch=compute_89,code=sm_89',  # Ada
    '-gencode=arch=compute_90,code=sm_90',  # Hopper
    '-gencode=arch=compute_100,code=sm_100', # Blackwell
    '--ptxas-options=-v',
    '-Xptxas=-O3',
    '-Xcompiler=-fPIC',
    '-std=c++20',
    '-D_GLIBCXX_USE_CXX11_ABI=0',
    '-DBLOTH_CUDA',
    '-DBLOTH_USE_TMA',
    '-DBLOTH_WARP_SPECIALIZATION',
]

CXX_FLAGS = [
    '-O3',
    '-std=c++20',
    '-fPIC',
    '-D_GLIBCXX_USE_CXX11_ABI=0',
    '-DBLOTH_CUDA',
]

# Include paths
INCLUDE_DIRS = [
    os.path.join(CUDA_HOME, 'include'),
    os.path.join(torch.utils.cpp_extension.CUDA_HOME or CUDA_HOME, 'include'),
    pybind11.get_include(),
    'bloth/kernels/cuda',
    'bloth/kernels/cuda/cutlass/include',
    'bloth/kernels/cuda/cutlass/tools/util/include',
]

# Library paths
LIBRARY_DIRS = [
    os.path.join(CUDA_HOME, 'lib64'),
    os.path.join(torch.utils.cpp_extension.CUDA_HOME or CUDA_HOME, 'lib64'),
]

# CUDA source files
CUDA_SOURCES = [
    'bloth/kernels/cuda/gemm_warp_specialized.cu',
    'bloth/kernels/cuda/tma_operations.cu',
    'bloth/kernels/cuda/fused_attention.cu',
    'bloth/kernels/cuda/layer_norm.cu',
    'bloth/kernels/cuda/fast_lora.cu',
    'bloth/kernels/cuda/quantization.cu',
    'bloth/kernels/cuda/softmax.cu',
    'bloth/kernels/cuda/rope_embedding.cu',
    'bloth/kernels/cuda/activations.cu',
    'bloth/kernels/cuda/memory_optim.cu',
    'bloth/kernels/cuda/python_bindings.cu',
]

# Check if we should build with CUTLASS
USE_CUTLASS = os.environ.get('BLOTH_USE_CUTLASS', '1') == '1'
if USE_CUTLASS:
    NVCC_FLAGS.extend(['-DBLOTH_USE_CUTLASS', '-DCUTLASS_ENABLE_TENSOR_CORE_MMA'])
    CXX_FLAGS.append('-DBLOTH_USE_CUTLASS')

# Check for FP8 support
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
    NVCC_FLAGS.append('-DBLOTH_FP8_SUPPORT')
    CXX_FLAGS.append('-DBLOTH_FP8_SUPPORT')

extensions = [
    CUDAExtension(
        name='bloth._C',
        sources=CUDA_SOURCES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        extra_compile_args={
            'cxx': CXX_FLAGS,
            'nvcc': NVCC_FLAGS,
        },
        libraries=['cudart', 'cublas', 'cublasLt'],
    ),
]

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bloth',
    version='0.1.0',
    author='Bloth Team',
    author_email='bloth@example.com',
    description='Ultra-Fast CUDA Kernels for LLM Training - Better than Unsloth',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/blothai/bloth',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'pybind11>=2.10.0',
        'numpy>=1.20.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.41.0',
        'peft>=0.4.0',
        'safetensors>=0.3.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    },
    zip_safe=False,
)
