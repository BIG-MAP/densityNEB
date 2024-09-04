import gc
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.calculators.vasp import VaspChargeDensity
from densityNEB import fill, torchneb
from densityNEB.interpolate_grid import Interpolator, calculate_grid_pos
from mayavi import mlab


def get_path_length(coords: np.ndarray):
    diffs = np.diff(coords, n=1, axis=0)
    lengths = np.linalg.norm(diffs, ord=2, axis=1)
    return np.sum(lengths)


def main():

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        filename2 = sys.argv[2]
    else:
        filename2 = None

    vasp_charge = VaspChargeDensity(filename)
    density = vasp_charge.chg[-1]
    atoms = vasp_charge.atoms[-1]
    grid_pos = calculate_grid_pos(
        density, np.zeros(3), atoms.get_cell()
    )

    interp = Interpolator(torch.tensor(density), atoms.get_cell())

    atomic_numbers = atoms.get_atomic_numbers().tolist()
    occurences = sorted(
        [(x, atomic_numbers.count(x)) for x in set(atomic_numbers)],
        key=lambda x: x[1], reverse=True
    )
    target_atoms = occurences[0][0]

    li_indices = np.nonzero(atoms.get_atomic_numbers() == target_atoms)[0]
    li_indices = li_indices[0:2]
    li_positions = atoms.get_positions()[
        atoms.get_atomic_numbers() == target_atoms
    ]
    li_positions = li_positions[0:2]

    li_average = np.mean(li_positions, axis=0, keepdims=True)
    direction_vectors = li_average-li_positions
    direction_vectors /= np.linalg.norm(
        direction_vectors, ord=2, axis=1, keepdims=True
    )

    init_neb_positions = li_positions*0.90 + 0.10*li_average

    # Plot initial neb trajectory
    x, y, z = init_neb_positions.T
    gx, gy, gz = grid_pos[..., 0], grid_pos[..., 1], grid_pos[..., 2]
    lower_bound = np.log10(max(np.min(density), 0.0) + 0.001)
    upper_bound = min(0, np.log10(np.max(density) - 0.001))
    contours = np.logspace(lower_bound, upper_bound, 5).tolist()
    mlab.contour3d(
        gx, gy, gz, density, contours=contours, vmax=1.0, vmin=0.01,
        opacity=1.0
    )
    mlab.plot3d(x, y, z)

    neb_optim_config = torchneb.OptimConfig(
        2000, torch.optim.SGD, {"lr": 0.2, "momentum": 0.1}
    )
    neb_config = torchneb.NEBConfig(optim_config=neb_optim_config)
    neb_config.spring_constant = float('inf')
    neb_config.insert_method = fill.equal
    neb_config.insert_args = {"count": 20}
    model = interp

    path_coords = torch.tensor(init_neb_positions)

    result = torchneb.neb({
        "path_coords": path_coords,
        "target_distances": torch.ones(2)
    }, model, neb_config, save_path_history=True)
    dense_vals = np.array(result["dense_vals"])
    final_path = result["path_coords"]

    init_length = get_path_length(init_neb_positions)
    final_path = result["path_coords"].numpy()
    path_length = get_path_length(final_path)

    print("Final path length=%f, init_length=%f" % (path_length, init_length))

    # Evaluate on other file
    if filename2 is not None:
        vasp_charge = VaspChargeDensity(filename2)
        density = vasp_charge.chg[-1]
        atoms = vasp_charge.atoms[-1]
        grid_pos = calculate_grid_pos(
            density, np.zeros(3), atoms.get_cell()
        )
        interp = Interpolator(torch.tensor(density), atoms.get_cell())
        dens2_vals = interp(final_path).cpu().numpy()
        plt.figure()
        plt.plot(np.linspace(0, 1, len(dens2_vals)), dens2_vals, '.-')

    # Plot final path
    x, y, z = final_path.T
    mlab.plot3d(x, y, z)
    neb_path = mlab.points3d(x, y, z, scale_factor=0.1, color=(1.0, 0., 0.))

    @mlab.animate(delay=10)
    def anim():
        for _ in range(10):
            for i, path in enumerate(result["path_history"]):
                npath = path.numpy()
                x, y, z = npath.T
                neb_path.mlab_source.set(x=x, y=y, z=z)
                gc.collect(generation=1)
                yield
    anim()

    mlab.show()

    plt.figure()
    plt.plot(np.linspace(0, 1, len(result["dense_vals"])), dense_vals, '.')
    plt.show()


if __name__ == "__main__":
    main()
