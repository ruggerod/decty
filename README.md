# decty
## Overview
This package implements custom changes to the popular 
[dtreeviz](https://github.com/parrt/dtreeviz) package,
with the main objective of offering an helper tool
for visualisation and exploratory analysis of decision trees
trained using [PySpark MLib](https://spark.apache.org/mllib/). 

The [main](https://github.com/ruggerod/decty/blob/main/scripts/main.ipynb)
notebook presents a quick example of the additions that decty 
implements upon dtreeviz core functionalities.

## Install
A [requirements](https://github.com/ruggerod/decty/blob/main/requirements.txt) 
file is provided to mimic the environment used during
development. This environment should be build locally
using the command `conda create --name <env> --file requirements.txt`. 
In the environment just created, the decty package can be 
then installed in development mode using e.g. `pip install -e .` from the 
main directory.