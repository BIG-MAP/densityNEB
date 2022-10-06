# densityNEB
Code for calculating the path of least resistance between two points in a scalar field using nudged elastic band (NEB).

The method is described in our paper [Accelerated Autonomous Workflow For Antiperovskite-based Solid State Electrolytes](https://dx.doi.org/10.21203/rs.3.rs-1780345/v1)

Parts of the code is based on [Pytorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB)

The two main components of the software is a module for linear interpolation on a grid `interpolate_grid.py` and a module for running NEB `torchneb.py`

# Installation
Clone the repository and install the python dependencies:

`pip install -r requirements.txt`

Only `numpy` and `torch` are strictly needed. The rest of the dependencies are only for running the examples.

# Running
You can run a simple toy example by running `torchneb.py` as a script

`python torchneb.py`

Another example script will visualize the NEB path finding between two Li atoms in a charge density file from VASP:

`python chgcar_neb_animation.py example_calculations/Li3SeF_Pm-3m/charge_density/CHGCAR`

The `chgcar_neb.py` file is included as an example on how to process multiple CHGCARs in a batch job.
