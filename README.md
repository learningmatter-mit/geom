# GEOM
GEOM: Energy-annotated molecular conformations
GEOM is a dataset with over 30 million molecular conformations annotated by energy and statistical weight for over 400,000 molecules. 

## Contents
This repository contains a jupyter notebook tutorial showing how to load the data and extract its contents.   

## Usage
This code was tested with the following dependencies:
python==3.7.5,
mgspack==1.0.0,
ipykernel==5.3.0

[MessagePack](https://msgpack.org/index.html) is a binary serialization format that allows you to exchange information among different languages, like JSON, but it is faster and more compact. You can create an anaconda environment to manage dependencies. First create an environment with Python 3.7.5:
```bash
conda upgrade conda
conda create -n geom python==3.7.5
```
Next activate the environment install `msgpack`:
```bash
conda activate geom
pip install msgpack==1.0.0 ipykernel==5.3.0
```
To ensure that the `geom` ernvironment is accessible through Jupyter, add the the `geom` display name:
```bash
python -m ipykernel install --user --name geom --display-name "Python [conda env:geom"]
```
