# lightning-pose-2024-nat-methods
This repository provides code to rerproduce the figures in the manuscript [Whiteway, Biderman et al, 2024](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1)

## Dependencies
The code has been tested on Ubuntu 18.04 and 22.04, using Python 3.10.
Required Python software packages are listed in [setup.py](https://github.com/themattinthehatt/lightning-pose-2024-nat-methods/blob/main/setup.py). 

## Installation
The installation takes about 7 min on a standard desktop computer. It is recommended to set up and activate a clean environment using conda or virtualenv, e.g.
```shell
virtualenv prior --python=python3.10
source prior/bin/activate
```

Then clone this repository and install it along with its dependencies
```shell
git clone https://github.com/themattinthehatt/lightning-pose-2024-nat-methods.git
cd lightning-pose-2024-nat-methods
pip install .
```

In a Python console, test if you can import functions:
```python
TODO
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
