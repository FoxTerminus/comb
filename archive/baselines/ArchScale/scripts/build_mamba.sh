#!/bin/bash
# Build mamba-ssm with correct C++ ABI for PyTorch 2.6 (old ABI)
set -e
export PATH="/data3/junhaohu/anaconda3/envs/samba/bin:$PATH"
export TMPDIR=/data3/junhaohu/tmp
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=12
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export TORCH_CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

echo "=== Building mamba-ssm ==="
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CXXFLAGS=$CXXFLAGS"

pip install --no-build-isolation --no-cache-dir --force-reinstall mamba-ssm==2.3.1 2>&1 | tail -20
echo "=== Done ==="
