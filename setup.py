from pathlib import Path
from setuptools import find_packages, setup


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
        raise RuntimeError('Unable to find version string.')


install_requires = [
    'matplotlib',
    'matplotlib-venn',
    'numpy',
    'opencv-python-headless',
    'pandas>=2.0.0',
    'pyarrow',
    'requests',
    'scikit-learn',
    'scipy',
    'seaborn',
]


setup(
    name='lightning_pose_plots',
    version=get_version(Path('lightning_pose_plots').joinpath('__init__.py')),
    license='MIT',
    description='Package for reproducing figures from Biderman, Whiteway et al 2024',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matt Whiteway',
    url='https://github.com/danbider/lightning-pose',
    packages=find_packages(exclude=['scratch']),  # same as name
    install_requires=install_requires,
)
