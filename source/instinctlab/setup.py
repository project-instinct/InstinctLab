"""Installation script for the 'instinctlab' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # Isaac Lab main source snapshot 8c9b995cb
    "isaaclab==41.0.1",
    "isaaclab-assets==2.1.0",
    "isaaclab-physx==9.0.0",
    "isaaclab-tasks==29.0.1",
    "isaaclab-visualizers==4.0.0",
    "isaaclab-newton==10.0.0",
    "instinct-rl==1.0.3",
    # Direct numerical/runtime dependencies
    "numpy>=2",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "gymnasium==1.3.0",
    "warp-lang==1.17.0",
    # InstinctLab-specific dependencies
    "pytorch_kinematics",
    "joblib",
    "debugpy",
    "snakeviz",
    "trimesh",
    # trimesh soft dependency for vector path / polygon handling in the trimesh terrain utilities
    "shapely>=2.0",
    "scikit-learn",
    "opencv-python",
    "onnxruntime>=1.20,<2",
    "pyvista",
]

# Installation operation
setup(
    name="instinctlab",
    packages=["instinctlab"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.2.0",
    ],
    zip_safe=False,
)
