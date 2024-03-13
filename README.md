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
If git is not recognized, [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

We recommend using [conda](https://docs.anaconda.com/free/anaconda/install/index.html) to create a 
new environment in which this package and its dependencies will be installed:
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

and install the package and dependencies:
```shell
pip install -e .
```

The installation takes about 5-10 min on a standard desktop computer.

In a Python console, test if you can import functions:
```python
from lightning_pose_plots import utilities
```

## Download figshare data
You will need to select a folder where the data and results are stored, referred to as `data_dir`
in this and the following command line calls 
(replace `/path/to/data` with a real path on your machine).
The following command will download the labeled data and the results, which is approximate 15GB.
This will take 10-20 minutes depending on your download speed.
```shell
python scripts/download_data.py --data_dir=/path/to/data
```

## Connecting to IBL database
In order to plot IBL results, you need to connect to the public IBL database to access example data.
The API, the Open Neurophysiology Environment (ONE), has already been installed with the requirements. 
If you have never used ONE, you can just establish the default database connection like this in a Python console. 
The first time you instantiate ONE you will have to enter the password (`international`): 
```python
from one.api import ONE
ONE.setup(silent=True)
one = ONE()
```

**NOTE**: if you have previously used ONE with a different database you might want to run this instead. 
Again, the first time you instantiate ONE you will have to enter the password (`international`):
```python
from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', make_default=False, silent=True)
one = ONE(base_url='https://openalyx.internationalbrainlab.org')
```

If you run into any issues refer to the [ONE documentation](https://int-brain-lab.github.io/ONE/index.html)

## Reproducing the figures
To automatically reproduce the main figures from the downloaded data, run
```console
python scripts/plot_figure.py --data_dir=/path/to/data --figure=all
```
This script will take approximately xx minutes to run.
Plots will be saved in a folder named `figures` inside the specified `data_dir`.

Individual figures can be reproduced by specifiying a number in {1, 2, 3, 4, 5} for the `figure`
argument:
```console
python scripts/plot_figure.py --data_dir=/path/to/data --figure=1
```

When reproducing individual figures, you may also specify a dataset for figures 3, 4 and 5:
```console
python scripts/plot_figure.py --data_dir=/path/to/data --figure=3 --dataset=mirror-fish
```

See `scripts/plot_figure.py` for more information on the command line arguments.
