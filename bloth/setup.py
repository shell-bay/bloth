"""
Bloth setup.py
Supports installing with or without CUTLASS for maximum performance.
Set BLOTH_USE_CUTLASS=1 before pip install for H100/B200 optimal kernels.
"""

import os
from setuptools import setup, find_packages

USE_CUTLASS = os.environ.get("BLOTH_USE_CUTLASS", "0") == "1"

setup(
    name             = "bloth",
    version          = "2.0.0",
    author           = "shell-bay",
    description      = "Ultra-Fast CUDA Kernel Library for LLM Training",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/shell-bay/bloth",
    license          = "Apache-2.0",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "torch>=2.0.0",
        "triton>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
    ],
    extras_require   = {
        "dev": ["pytest", "black", "isort"],
        "cutlass": [],  # CUTLASS is cloned separately — see README
        "full": ["bitsandbytes>=0.41.0", "datasets", "accelerate"],
    },
    classifiers      = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
