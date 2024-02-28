import sys
from pathlib import Path

from setuptools import find_packages, setup

CURRENT_DIRECTORY = Path(__file__).parent.absolute()

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 8)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
==========================
Unsupported Python version
==========================
This version of brainwide encodeco requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(
            *(REQUIRED_PYTHON + CURRENT_PYTHON)
        )
    )
    sys.exit(1)

with open("README.md", "r") as f:
    long_description = f.read()


def read(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


install_requires = [
    "ibllib",
    "matplotlib",
    "ONE-api",
    "opencv-python",
    "pandas>=2.0.0",
    "seaborn",
]


setup(
    name="lightning_pose_plots",
    version=get_version(Path("lightning_pose_plots").joinpath("__init__.py")),
    python_requires=">={}.{}".format(*REQUIRED_PYTHON),
    license="MIT",
    description="Package for reproducing figures from Biderman, Whiteway et al 2024",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matt Whiteway",
    url="https://github.com/danbider/lightning-pose",
    packages=find_packages(exclude=["scratch"]),  # same as name
    install_requires=install_requires,
)
