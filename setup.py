#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from setuptools import Extension, setup
from torch.utils import cpp_extension

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")

def write_version_py():
    """Writes the version to version.py, which is used by the package."""
    with open(os.path.join(os.path.dirname(__file__), "fairseq", "version.txt")) as f:
        version = f.read().strip()
    with open(os.path.join(os.path.dirname(__file__), "fairseq", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version

version = write_version_py()

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

class NumpyExtension(Extension):
    """Custom Extension class to handle numpy includes."""
    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs

extensions = [
    Extension(
        "fairseq.libbleu",
        sources=[
            "fairseq/clib/libbleu/libbleu.cpp",
            "fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.token_block_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    cpp_extension.CppExtension(
        "fairseq.libbase",
        sources=["fairseq/clib/libbase/balanced_assignment.cpp"],
    ),
    cpp_extension.CppExtension(
        "fairseq.libnat",
        sources=["fairseq/clib/libnat/edit_dist.cpp"],
    ),
    cpp_extension.CppExtension(
        "alignment_train_cpu_binding",
        sources=["examples/operators/alignment_train_cpu.cpp"],
    ),
]

if "CUDA_HOME" in os.environ:
    extensions.extend([
        cpp_extension.CppExtension(
            "fairseq.libnat_cuda",
            sources=[
                "fairseq/clib/libnat_cuda/edit_dist.cu",
                "fairseq/clib/libnat_cuda/binding.cpp",
            ],
        ),
        cpp_extension.CppExtension(
            "fairseq.ngram_repeat_block_cuda",
            sources=[
                "fairseq/clib/cuda/ngram_repeat_block_cuda.cpp",
                "fairseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu",
            ],
        ),
        cpp_extension.CppExtension(
            "alignment_train_cuda_binding",
            sources=[
                "examples/operators/alignment_train_kernel.cu",
                "examples/operators/alignment_train_cuda.cpp",
            ],
        ),
    ])

cmdclass = {"build_ext": cpp_extension.BuildExtension}

if "READTHEDOCS" in os.environ:
    extensions = []

setup(
    version=version,
    ext_modules=extensions,
    cmdclass=cmdclass,
)
