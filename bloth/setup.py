"""
Bloth v1.0 setup.py
Fixed issues from Colab:
  - README is read safely (won't crash if path is wrong)
  - No PEP 517 editable-install conflicts
  - Compatible with: pip install .  AND  python setup.py install
"""

import os
from setuptools import setup, find_packages

# Safe README read — won't crash regardless of working directory
_here    = os.path.dirname(os.path.abspath(__file__))
_readme  = os.path.join(_here, "README.md")
long_desc = open(_readme, encoding="utf-8").read() if os.path.exists(_readme) else ""

setup(
    name                          = "bloth",
    version                       = "1.0.0",
    author                        = "shell-bay",
    author_email                  = "",
    description                   = ("Ultra-Fast CUDA Kernel Library for LLM Training — "
                                     "Outperforms Unsloth via Hyper-Fused Triton Kernels"),
    long_description              = long_desc,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/shell-bay/bloth",
    license                       = "Apache-2.0",
    packages                      = find_packages(exclude=["tests*", "examples*", "docs*"]),
    python_requires               = ">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
        # Pin compatible versions — fixes the trl/transformers mismatch from Colab
        "transformers>=4.46.3",
        "peft>=0.7.0",
        "trl>=0.11.4",
    ],
    extras_require={
        "full": [
            "bitsandbytes>=0.41.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "scipy",
        ],
        "dev": ["pytest>=7.0", "black", "isort"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
