# lightning-pose-2024-nat-methods
This repository provides code to rerproduce the figures in the manuscript [Biderman, Whiteway et al, 2024](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1).

## Dependencies
The code has been tested on Ubuntu 18.04 and 22.04, using Python 3.10.
Required Python software packages are listed in [setup.py](https://github.com/themattinthehatt/lightning-pose-2024-nat-methods/blob/main/setup.py). 

## Installation with conda

First, ensure git is installed:
```shell
git --version
````
If ‘git’ is not recognized, [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

We recommend using [conda](https://docs.anaconda.com/free/anaconda/install/index.html) to create a new environment in which this package and its dependencies will be installed:
```shell
conda create --name lpplots python=3.10
```

Activate the new environment:
```shell
conda activate lpplots
```
Make sure you are in the activated environment during the following steps.

Move into the directory where you want to place the repository folder, and then download it from GitHub:
```shell
cd <SOME_FOLDER>
git clone https://github.com/themattinthehatt/lightning-pose-2024-nat-methods.git
```

Then move into the newly-created repository folder:
```shell
cd lightning-pose-2024-nat-methods
```

and install dependencies using one of the lines below that suits your needs best:
```shell
pip install -e .
```

The installation takes about 7 min on a standard desktop computer

In a Python console, test if you can import functions:
```python
from lightning_pose_plots import utilities
```

## Download figshare data
```shell
python scripts/download_data.py --data_dir=/home/mattw/data
```

## Connecting to IBL database
In order to run the example code or the tests, you need to connect to the public IBL database to access example data.
Our API, the Open Neurophysiology Environment (ONE) has already been installed with the requirements. 
If you have never used ONE, you can just establish the default database connection like this in a Python console. 
The first time you instantiate ONE you will have to enter the password (`international`) 
```python
from one.api import ONE
ONE.setup(silent=True)
one = ONE()
```

**NOTE**: if you have previously used ONE with a different database you might want to run this instead. Again, the 
first time you instantiate ONE you will have to enter the password (`international`)
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', make_default=False, silent=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
```

If you run into any issues refer to the [ONE documentation](https://int-brain-lab.github.io/ONE/index.html)

## Running example code
TODO
The data is automatically downloaded from the public IBL database, provided that the above ONE setup has been performed.
