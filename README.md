# GEOM
GEOM: Energy-annotated molecular conformations
GEOM is a dataset with over 30 million molecular conformations annotated by energy and statistical weight for over 400,000 molecules. 

## Contents
This repository contains [Jupyter notebook tutorials](https://github.com/learningmatter-mit/geom/blob/master/tutorials) showing how to load the data and perform analysis.

## Usage
This code was tested with the following dependencies:
python==3.7.5,
mgspack==1.0.0,
ipykernel==5.3.0,
rdkit==2020.03.2.0,
matplotlib==3.2.1.


You can create an anaconda environment to manage dependencies. First create an environment with Python, RDKit, and Matplotlib:
```bash
conda upgrade conda
conda create -n geom python==3.7.5 rdkit==2020.03.2.0 matplotlib==3.2.1 -c rdkit -c conda-forge 

```
Next activate the environment install `msgpack` and `ipykernel`:
```bash
conda activate geom
pip install msgpack==1.0.0 ipykernel==5.3.0
```
To ensure that the `geom` environment is accessible through Jupyter, add the the `geom` display name:
```bash
python -m ipykernel install --user --name geom --display-name "Python [conda env:geom"]
```

## Accessing the data

### Language-agnostic data

The datasets are available at [here](https://www.dropbox.com/sh/1aptf9fi8kyrzg6/AABQ4F7dpl4tQ_pGCf2izd7Ca?dl=0). There are four datasets that can be loaded by any programming language. They are `drugs_crude.msgpack.tar.gz`, `drugs_featurized.msgpack.tar.gz`, `qm9_crude.msgpack.tar.gz`, `qm9_featurized.msgpack.tar.gz`. The [tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/01_loading_data.ipynb) gives instructions for extracting the files and loading their content using MessagePack. [MessagePack](https://msgpack.org/index.html) is a binary serialization format that allows you to exchange information among different languages, like JSON, but it is faster and more compact. 

### Python-specific data

The featurized files contain bond and atom features as lists of dictionaries and are quite large. If you are using Python, it is far more convenient to use the folder `rdkit_folder.tar`. This folder contains files in which the conformer coordinates are replaced by RDKit `mol` objects. These objects contain both the coordinates and all the connectivity information contained in the `featurized` files, but use far less disk space. Moreover, with RDKit you can generate your own 2D and 3D descriptors in a very straightforward way. The [RDKit tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb) shows how to load the RDKit files, visualize conformers, generate additional descriptors, and export to PDB. If you are not familiar with RDKit, you can get started at the [RDKit home page](https://www.rdkit.org/docs/index.html).

Finally, you may want to analyze only a few molecules based on certain properties (e.g., load 200 molecules that bind SARS-CoV 3CL protease, and 1000 that do not). However, you may not want to first load *all* molecules and *then* filter by properties. In this case you can load the files `rdkit_folder/{drugs,qm9}_summary.json`, which contain all the summary statistics for each molecule, but exclude conformer information. You can use these lightweight files to decide which molecules to load, and then load their RDKit pickle files one-by-one. This too is described in the [RDKit tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb).





