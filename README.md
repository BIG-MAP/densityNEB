![Logo of densityNEB software for calculating path of least resistance between two points in a scalar field using nudged elastic band](logo/logo.svg?raw=true "densityNEB logo")

# densityNEB
Code for calculating the path of least resistance between two points in a scalar field using nudged elastic band (NEB).

The method is described in our paper [Accelerated Autonomous Workflow For Antiperovskite-based Solid State Electrolytes](https://dx.doi.org/10.21203/rs.3.rs-1780345/v1).

Parts of the code is based on [Pytorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB).

The two main components of the software is a module for linear interpolation on a grid `interpolate_grid.py` and a module for running NEB `torchneb.py`.

# Installation
Clone the repository and install the code as a python package by running the following in the cloned directory:

`pip install .`

> If you want to be able to run the examples with an interactive window showing the path on the charge density, install it with `pip install .[viewer]`. This will install `mayavi` and `PyQt5`. If using these with an X-server, make sure not to use Xming, since it's generally unsupported by `PyQt5`.

Only `numpy` and `torch` are strictly needed. The rest of the dependencies are only for running the examples.

# Running
You can run a simple toy example by running `torchneb.py` as a script:

`python src/densityNEB/torchneb.py`

Another example script will visualize the NEB path finding between two Li atoms in a charge density file from VASP:

`python examples/chgcar_neb_animation.py examples/calculations/Li3SeF_Pm-3m/charge_density/CHGCAR`

The `chgcar_neb.py` file is included as an example on how to process multiple CHGCARs in a batch job.

# Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 957189.
