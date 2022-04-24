# GEOM: Energy-annotated molecular conformations 

GEOM is a dataset with 37 million molecular conformations annotated by energy and statistical weight for over 450,000 molecules. If you use this data, please cite

Axelrod, S. and Gómez-Bombarelli, R. GEOM, energy-annotated molecular conformations for property prediction and molecular generation. *Scientific Data* **9**, 185 (2022). https://doi.org/10.1038/s41597-022-01288-4

Bibtex format:

```
@article{axelrod2022geom,
	 author = {Axelrod, Simon and G{\'o}mez-Bombarelli, Rafael},
	 doi = {10.1038/s41597-022-01288-4},
	 journal = {Scientific Data},
	 number = {1},
	 pages = {185},
	 title = {GEOM, energy-annotated molecular conformations for property prediction and molecular generation},
	 url = {https://doi.org/10.1038/s41597-022-01288-4},
	 volume = {9},
	 year = {2022}
}
```


## Contents
This repository contains [Jupyter notebook tutorials](https://github.com/learningmatter-mit/geom/blob/master/tutorials) showing how to load the data and perform analysis.

## Usage
This code was tested with the following dependencies:
```
python==3.7.10,
mgspack==1.0.3,
ipykernel==6.9.0,
rdkit==2020.09.1,
matplotlib==3.2.2,
e3fp==1.2.3
tqdm==4.62.3
ase==3.22.1
pytorch==1.4.0 
scikit-learn==1.0.2 
nglview==3.0.3
sympy==1.9 
networkx==2.6.3
```

You can create an [anaconda](https://conda.io/docs/index.html) environment to manage dependencies. You can learn more about managing anaconda environments by reading [this page](http://conda.pydata.org/). First create an environment with Python, RDKit, and Matplotlib:

```bash
conda upgrade conda
conda create -n geom python==3.7.10 rdkit==2020.09.1 matplotlib==3.2.2 tqdm==4.62.3 ase==3.22.1 pytorch==1.4.0 scikit-learn==1.0.2 -c rdkit -c conda-forge -c pytorch -c anaconda

```
Next activate the environment and install `pip` packages:
```bash
conda activate geom
pip install msgpack==1.0.3 ipykernel==6.9.0 e3fp==1.2.3 nglview==3.0.3 sympy==1.9 networkx==2.6.3
```
To ensure that the `geom` environment is accessible through Jupyter, add the the `geom` display name:
```bash
python -m ipykernel install --user --name geom --display-name "Python [conda env:geom"]
```

## Accessing the data

### Language-agnostic data

The datasets are available [here](https://doi.org/10.7910/DVN/JNGTDF). There are four datasets that can be loaded by any programming language. They are `drugs_crude.msgpack.tar.gz`, `drugs_featurized.msgpack.tar.gz`, `qm9_crude.msgpack.tar.gz`, `qm9_featurized.msgpack.tar.gz`. The [first tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/01_loading_data.ipynb) gives instructions for extracting the files and loading their content using [MessagePack](https://msgpack.org/index.html). MessagePack is a binary serialization format that allows you to exchange information among different languages, like JSON, but it is faster and more compact. 

**Note: we will be updating the Python-specific data below as we add new molecules to GEOM, but we will not be updating the MessagePack files. If you want the new molecules but you don't use Python, please reach out to us and we will be happy to help.**

### Python-specific data

The featurized files contain bond and atom features as lists of dictionaries and are quite large. If you are using Python, it is far more convenient to use the folder `rdkit_folder.tar.gz`. This folder contains files in which the conformer coordinates are replaced by RDKit `mol` objects. These objects contain both the coordinates and all the connectivity information contained in the `featurized` files, but use far less disk space. Moreover, with RDKit you can generate your own 2D and 3D descriptors in a very straightforward way. The [RDKit tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb) shows how to load the RDKit files, visualize conformers, generate additional descriptors, and export to PDB. If you are not familiar with RDKit, you can get started at the [RDKit home page](https://www.rdkit.org/docs/index.html).

Finally, you may want to analyze only a few molecules based on certain properties (e.g., load 200 molecules that bind SARS-CoV 3CL protease, and 1000 that do not). However, you may not want to first load *all* molecules and *then* filter by properties. In this case you can load the files `rdkit_folder/{drugs,qm9}_summary.json`, which contain all the summary statistics for each molecule, but exclude conformer information. You can use these lightweight files to decide which molecules to load, and then load their RDKit pickle files one-by-one. This, too, is described in the [RDKit tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/02_loading_rdkit_mols.ipynb).


### Data updates
1. **New drug-like molecules** (Feb. 1, 2021). We have updated the GEOM dataset since our paper was first posted on the ArXiv, adding about 13,000 new drug-like molecules, including about 6,000 with SARS-CoV-2 data. To make sure that you have the latest version of the data, please see the `README` file in the [data folder](https://doi.org/10.7910/DVN/JNGTDF). This explains how you can generate a checksum and confirm that it matches ours. If it matches then you have the latest version of the data. If not you will want to download it again.

2. **MoleculeNet** (Feb. 9, 2022). We have added over 16,000 species from the MoleculeNet dataset. These species have data related to physical chemistry, physiology, and biophysics. These results can be found with the rest of the data [here](https://doi.org/10.7910/DVN/JNGTDF). The data contains CREST ensembles for the species, together with Hessian data and high-accuracy DFT results for the BACE subset. To load this data, see our [MoleculeNet tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/03_loading_molecule_net.ipynb). To see how to compare results from different levels of theory, see our [comparison tutorial](https://github.com/learningmatter-mit/geom/blob/master/tutorials/04_comparing_ensembles.ipynb). To use these tutorials, make sure to update your `geom` environment with the new packages and versions given above.


## Training machine learning models
To train conformer-based machine learning models on the GEOM data, you can use the [Neural Force Field](https://github.com/learningmatter-mit/NeuralForceField) repository. In particular, you can navigate to the Cp3D tutorial to see how the models are created and trained, and to `scripts/cp3d` to learn how to run training scripts.

