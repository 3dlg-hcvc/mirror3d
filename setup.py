#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="mirror3d",
    version="1.0",
    author="3dlg-hcvc",
    url="https://3dlg-hcvc.github.io/mirror3d/",
    description="Code for Mirror3D",
    packages=find_packages(exclude=("docs", "script")),
    install_requires=["torchvision>=0.4", "fvcore", "matplotlib", \
        "pycocotools","tabulate","torch","colorlog","dill","easydict", \
        "efficientnet_pytorch","numpy","Pillow","PyYAML","scipy","six", \
        "tensorboardX","torchvision","tqdm","opencv-python", \
        "scikit-image","sympy","beautifulsoup4","open3d==0.12.0"],
)
