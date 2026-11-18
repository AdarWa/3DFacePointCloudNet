"""Build the PointNet++ CUDA ops (``pointnet2._ext``) and install the package.

    pip install -r requirements.txt
    python setup.py build_ext --inplace     # builds pointnet2/_ext*.so in place
    # or: pip install -e . --no-build-isolation

Requires a CUDA toolkit (nvcc) matching the CUDA version PyTorch was built with.
Set TORCH_CUDA_ARCH_LIST (e.g. "7.5;8.6;8.9") to choose GPU architectures, which
is needed when building on a machine without a visible GPU.
"""
import glob
import os

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyTorch must be installed before building: pip install torch") from e

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_SRC = os.path.join("pointnet2", "_ext-src")
INCLUDE_DIR = os.path.join(ROOT, EXT_SRC, "include")

sources = sorted(glob.glob(os.path.join(EXT_SRC, "src", "*.cpp")) +
                 glob.glob(os.path.join(EXT_SRC, "src", "*.cu")))

setup(
    name="pointnet2",
    version="1.0.0",
    description="PointNet++ for 3D face recognition from point clouds",
    packages=find_packages(include=["pointnet2", "pointnet2.*"]),
    python_requires=">=3.8",
    install_requires=["torch", "torchvision", "numpy", "scikit-learn", "tqdm"],
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=sources,
            include_dirs=[INCLUDE_DIR],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
